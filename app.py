import os
import sys
import time
import pickle
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# =============================================================================
# 1. MODEL ARCHITECTURES (Exact match with Kaggle Training Notebook)
# =============================================================================

class SiameseLSTM(nn.Module):
    def __init__(self, input_size: int = 3, hidden: int = 128, embed_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.embed = nn.Linear(hidden, embed_dim)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.embed(h_n[-1])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return self.forward_once(x1), self.forward_once(x2)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = 3, hidden: int = 128, num_classes: int = 300):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(h))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 64, max_len: int = 15):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int = 3, d_model: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 300):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=15)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))


# =============================================================================
# 2. APP CONFIGURATION & ARTIFACT LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "keystroke_models")

app = FastAPI(title="Keystroke Dynamics Biometrics")

# Mount static folder for CSS and JS
static_path = os.path.join(BASE_DIR, "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model & artifact containers
siamese_model: SiameseLSTM = None
lstm_model: LSTMClassifier = None
transformer_model: TransformerClassifier = None
scaler_full = None
scaler_id = None
label_encoder_id = None
demo_reference_samples = {}

# In-memory user profiles: stores both live enrolled users and benchmark users
PROFILES: Dict[str, Dict[str, Any]] = {}

# Hyperparameters matching training
SEQ_LEN = 50       # Siamese LSTM sequence length
ID_SEQ_LEN = 15    # Classifier sequence length
DEFAULT_THRESHOLD = 0.75  # Distance threshold for verification (based on ~EER 0.23)

def load_artifacts():
    global siamese_model, lstm_model, transformer_model
    global scaler_full, scaler_id, label_encoder_id, demo_reference_samples, PROFILES

    print("==================================================")
    print("Loading Keystroke Dynamics Models & Checkpoints...")
    print(f"Target Device: {DEVICE}")

    # 1. Load Scalers
    with open(os.path.join(MODELS_DIR, "scaler_full.pkl"), "rb") as f:
        scaler_full = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler_id.pkl"), "rb") as f:
        scaler_id = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "label_encoder_id.pkl"), "rb") as f:
        label_encoder_id = pickle.load(f)
    print(f"[OK] Scalers & Label Encoder loaded ({len(label_encoder_id.classes_)} classes)")

    # 2. Siamese Model
    siamese_model = SiameseLSTM(input_size=3, hidden=128, embed_dim=32).to(DEVICE)
    siamese_state = torch.load(os.path.join(MODELS_DIR, "siamese_model.pt"), map_location=DEVICE)
    if any(k.startswith("module.") for k in siamese_state.keys()):
        siamese_state = {k.replace("module.", ""): v for k, v in siamese_state.items()}
    siamese_model.load_state_dict(siamese_state)
    siamese_model.eval()
    print("[OK] SiameseLSTM (32-dim embedding) loaded")

    # 3. LSTM Classifier
    lstm_model = LSTMClassifier(input_size=3, hidden=128, num_classes=len(label_encoder_id.classes_)).to(DEVICE)
    lstm_state = torch.load(os.path.join(MODELS_DIR, "lstm_model.pt"), map_location=DEVICE)
    if any(k.startswith("module.") for k in lstm_state.keys()):
        lstm_state = {k.replace("module.", ""): v for k, v in lstm_state.items()}
    lstm_model.load_state_dict(lstm_state)
    lstm_model.eval()
    print("[OK] Bidirectional LSTMClassifier loaded")

    # 4. Transformer Classifier
    transformer_model = TransformerClassifier(
        input_size=3, d_model=64, nhead=4, num_layers=2, num_classes=len(label_encoder_id.classes_)
    ).to(DEVICE)
    trans_state = torch.load(os.path.join(MODELS_DIR, "transformer_model.pt"), map_location=DEVICE)
    if any(k.startswith("module.") for k in trans_state.keys()):
        trans_state = {k.replace("module.", ""): v for k, v in trans_state.items()}
    transformer_model.load_state_dict(trans_state)
    transformer_model.eval()
    print("[OK] TransformerClassifier loaded")

    # 5. Load Demo Reference Benchmark Profiles
    demo_path = os.path.join(MODELS_DIR, "demo_reference_samples.pkl")
    if os.path.exists(demo_path):
        with open(demo_path, "rb") as f:
            demo_reference_samples = pickle.load(f)

        with torch.no_grad():
            for uid, samples in demo_reference_samples.items():
                t_samples = torch.tensor(samples, dtype=torch.float32).to(DEVICE)
                embs = siamese_model.forward_once(t_samples).cpu().numpy()
                centroid = np.mean(embs, axis=0)
                PROFILES[f"Benchmark User {uid}"] = {
                    "id": f"benchmark_{uid}",
                    "name": f"Benchmark User {uid}",
                    "is_benchmark": True,
                    "centroid": centroid.tolist(),
                    "num_samples": len(samples),
                    "created_at": "Pre-trained benchmark"
                }
        print(f"[OK] Initialized {len(PROFILES)} benchmark profiles")
    print("==================================================")


# Pre-load models at startup
@app.on_event("startup")
async def startup_event():
    load_artifacts()


# =============================================================================
# 3. FEATURE EXTRACTION PIPELINE
# =============================================================================

def extract_features_from_events(events: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extracts [Hold Time, Down-Down Time, Up-Down Time] from raw keystroke timestamps.
    Returns: np.ndarray of shape (N, 3) where N = len(events)
    """
    if len(events) < 2:
        return np.empty((0, 3), dtype=np.float32)

    features = []
    for i in range(len(events)):
        p_curr = float(events[i]["press_time"])
        r_curr = float(events[i]["release_time"])

        # 1. Hold Time = release(i) - press(i)
        h = max(1.0, r_curr - p_curr)

        # 2. Down-Down Latency = press(i) - press(i-1)
        if i == 0:
            dd = 0.0
        else:
            p_prev = float(events[i - 1]["press_time"])
            dd = max(0.0, p_curr - p_prev)

        # 3. Up-Down Flight Time = press(i) - release(i-1)
        if i == 0:
            ud = 0.0
        else:
            r_prev = float(events[i - 1]["release_time"])
            ud = p_curr - r_prev

        features.append([h, dd, ud])

    return np.array(features, dtype=np.float32)


def prepare_siamese_tensor(feats: np.ndarray) -> torch.Tensor:
    """Standardizes with scaler_full and pads/truncates to SEQ_LEN=50."""
    scaled = scaler_full.transform(feats)
    n = len(scaled)

    if n >= SEQ_LEN:
        seq = scaled[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - n, 3), dtype=np.float32)
        seq = np.vstack([scaled, pad])

    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def prepare_classifier_tensor(feats: np.ndarray) -> torch.Tensor:
    """Standardizes with scaler_id and pads/truncates to ID_SEQ_LEN=15."""
    scaled = scaler_id.transform(feats)
    n = len(scaled)

    if n >= ID_SEQ_LEN:
        seq = scaled[:ID_SEQ_LEN]
    else:
        pad = np.zeros((ID_SEQ_LEN - n, 3), dtype=np.float32)
        seq = np.vstack([scaled, pad])

    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)


# =============================================================================
# 4. REST API ROUTES (FastAPI Async Endpoints)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serves the main interactive dashboard."""
    template_file = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_file, "r", encoding="utf-8") as f:
        content = f.read()
    response = HTMLResponse(content=content)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/model_info")
async def get_model_info():
    """Returns architectural parameters, dataset info, and active registered profiles."""
    return JSONResponse(content={
        "dataset": "Aalto University 136M Keystrokes Dataset",
        "device": str(DEVICE),
        "siamese_model": {
            "name": "Siamese LSTM (Twin Branches)",
            "input_dim": 3,
            "hidden_dim": 128,
            "embed_dim": 32,
            "sequence_length": SEQ_LEN,
            "eer": 0.2336,
            "accuracy_at_eer": "76.68%",
            "default_threshold": DEFAULT_THRESHOLD
        },
        "transformer_model": {
            "name": "Transformer Encoder Classifier",
            "d_model": 64,
            "heads": 4,
            "layers": 2,
            "sequence_length": ID_SEQ_LEN,
            "top1_accuracy": "51.25%",
            "num_classes": len(label_encoder_id.classes_) if label_encoder_id else 0
        },
        "lstm_model": {
            "name": "Bidirectional LSTM Classifier",
            "hidden_dim": 128,
            "bidirectional": True,
            "sequence_length": ID_SEQ_LEN,
            "top1_accuracy": "36.22%",
            "num_classes": len(label_encoder_id.classes_) if label_encoder_id else 0
        },
        "registered_profiles": [
            {
                "id": p["id"],
                "name": p["name"],
                "is_benchmark": p.get("is_benchmark", False),
                "num_samples": p.get("num_samples", 0),
                "created_at": p.get("created_at", "")
            }
            for p in PROFILES.values()
        ]
    })


@app.post("/api/enroll")
async def enroll_user(request: Request):
    """Enrolls a new user profile by averaging embeddings of 3-10 typed samples."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid JSON payload."})

    user_name = data.get("user_name", "").strip() or "Enrolled User"
    samples_data = data.get("samples", [])

    if not samples_data or len(samples_data) < 3:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": "At least 3 typing samples are required to create a stable reference profile (5-10 recommended)."
        })

    embeddings = []
    all_holds, all_dds, all_uds = [], [], []

    with torch.no_grad():
        for sample in samples_data:
            feats = extract_features_from_events(sample)
            if len(feats) < 3:
                continue
            all_holds.extend(feats[:, 0].tolist())
            all_dds.extend(feats[1:, 1].tolist())
            all_uds.extend(feats[1:, 2].tolist())

            tensor = prepare_siamese_tensor(feats)
            emb = siamese_model.forward_once(tensor).cpu().numpy().squeeze(0)
            embeddings.append(emb)

    if len(embeddings) < 2:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": "Not enough valid keystrokes captured across samples."
        })

    centroid = np.mean(embeddings, axis=0)

    distances_to_centroid = [float(np.linalg.norm(e - centroid)) for e in embeddings]
    mean_consistency = float(np.mean(distances_to_centroid))

    profile_key = f"user_{user_name.lower().replace(' ', '_')}"
    PROFILES[profile_key] = {
        "id": profile_key,
        "name": user_name,
        "is_benchmark": False,
        "centroid": centroid.tolist(),
        "num_samples": len(embeddings),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "mean_hold_ms": round(float(np.mean(all_holds)), 1) if all_holds else 0,
            "mean_flight_ms": round(float(np.mean(all_uds)), 1) if all_uds else 0,
            "mean_consistency_dist": round(mean_consistency, 3)
        }
    }

    return JSONResponse(content={
        "success": True,
        "profile_key": profile_key,
        "profile_name": user_name,
        "num_samples_used": len(embeddings),
        "consistency_score": round(max(0, 100 - mean_consistency * 60), 1),
        "stats": PROFILES[profile_key]["stats"]
    })


@app.post("/api/verify")
async def verify_keystroke(request: Request):
    """Verifies an incoming keystroke sample against a selected profile."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid JSON payload."})

    profile_key = data.get("profile_key", "")
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
    events = data.get("events", [])

    if not events or len(events) < 3:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": "Too few keystrokes (minimum 3 required, 10-15 recommended)."
        })

    if profile_key not in PROFILES:
        if PROFILES:
            profile_key = list(PROFILES.keys())[0]
        else:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "No reference profile found. Please enroll or select a benchmark profile first."
            })

    target_profile = PROFILES[profile_key]
    ref_centroid = np.array(target_profile["centroid"], dtype=np.float32)

    feats = extract_features_from_events(events)
    if len(feats) < 2:
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid timing sequence."})

    hold_times = feats[:, 0]
    flight_times = feats[1:, 2] if len(feats) > 1 else feats[:, 2]
    mean_hold = round(float(np.mean(hold_times)), 1)
    mean_flight = round(float(np.mean(flight_times)), 1)

    total_duration_sec = (float(events[-1]["release_time"]) - float(events[0]["press_time"])) / 1000.0
    wpm = round((len(events) / 5.0) / (total_duration_sec / 60.0), 1) if total_duration_sec > 0.3 else 0.0

    with torch.no_grad():
        tensor = prepare_siamese_tensor(feats)
        test_emb = siamese_model.forward_once(tensor).cpu().numpy().squeeze(0)

    distance = float(np.linalg.norm(test_emb - ref_centroid))
    similarity_score = max(0.0, min(100.0, (1.0 - (distance / (threshold * 2.2))) * 100.0))

    is_verified = bool(distance < threshold)
    verdict = "Verified ✅" if is_verified else "Impostor ❌"
    status_text = "Access Granted — Typing dynamics match the registered behavioral profile." if is_verified else "Access Denied — Significant divergence in rhythm and cadence detected."

    return JSONResponse(content={
        "success": True,
        "verdict": verdict,
        "is_verified": is_verified,
        "status_text": status_text,
        "distance": round(distance, 4),
        "threshold": round(threshold, 4),
        "similarity_score": round(similarity_score, 1),
        "target_profile": target_profile["name"],
        "timing_stats": {
            "keystroke_count": len(events),
            "mean_hold_ms": mean_hold,
            "mean_flight_ms": mean_flight,
            "wpm": wpm,
            "duration_sec": round(total_duration_sec, 2)
        }
    })


@app.post("/api/identify")
async def identify_user(request: Request):
    """Classifies an incoming keystroke sample across 300 registered classes."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid JSON payload."})

    events = data.get("events", [])

    if not events or len(events) < 3:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": "Too few keystrokes (minimum 3 required)."
        })

    feats = extract_features_from_events(events)
    tensor = prepare_classifier_tensor(feats)

    with torch.no_grad():
        trans_logits = transformer_model(tensor)
        trans_probs = F.softmax(trans_logits, dim=1).cpu().numpy().squeeze(0)

        lstm_logits = lstm_model(tensor)
        lstm_probs = F.softmax(lstm_logits, dim=1).cpu().numpy().squeeze(0)

    def get_top5(probs):
        top_indices = np.argsort(probs)[::-1][:5]
        top5_list = []
        for rank, idx in enumerate(top_indices, 1):
            uid = str(label_encoder_id.classes_[idx])
            prob_percent = round(float(probs[idx]) * 100.0, 2)
            top5_list.append({
                "rank": rank,
                "user_id": uid,
                "probability": prob_percent
            })
        return top5_list

    transformer_top5 = get_top5(trans_probs)
    lstm_top5 = get_top5(lstm_probs)

    return JSONResponse(content={
        "success": True,
        "transformer_top5": transformer_top5,
        "lstm_top5": lstm_top5,
        "total_classes": len(label_encoder_id.classes_)
    })


# =============================================================================
# 5. ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"\n>>> Keystroke Dynamics Biometric Web App Running on http://0.0.0.0:{port}")
    print("Press CTRL+C in terminal to stop server.\n")
    uvicorn.run(app, host="0.0.0.0", port=port)

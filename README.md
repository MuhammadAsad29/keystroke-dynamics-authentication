# 🔐 Keystroke Dynamics Behavioral Biometric Authentication & Identification System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-Aalto%20136M%20Keystrokes-blue)](https://userinterfaces.aalto.fi/136Mkeystrokes/)
[![Deployed on Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)](https://render.com)

A deep learning behavioral biometric system that authenticates users based on the physical timing rhythm of their typing. Trained on the massive **Aalto University 136 Million Keystroke Dataset** across **167,000+ real-world typists**, this repository implements both **1:1 Identity Verification (Siamese LSTM with Contrastive Loss)** and **1:N User Identification (Transformer Encoder vs. Bidirectional LSTM)**, paired with a real-time, interactive, glassmorphic web dashboard.

---

## 📑 Table of Contents

- [Key Highlights](#-key-highlights)
- [Behavioral Biometrics & Feature Engineering](#-behavioral-biometrics--feature-engineering)
- [Dataset Preprocessing & Augmentation](#-dataset-preprocessing--augmentation)
- [Deep Learning Architectures](#-deep-learning-architectures)
  - [1. Siamese LSTM (1:1 Verification)](#1-siamese-lstm-11-verification)
  - [2. Transformer Encoder Classifier (1:N Identification)](#2-transformer-encoder-classifier-1n-identification)
  - [3. Bidirectional LSTM Classifier (1:N Identification)](#3-bidirectional-lstm-classifier-1n-identification)
- [Training Hyperparameters & Experimental Setup](#-training-hyperparameters--experimental-setup)
- [Evaluation & Benchmark Results](#-evaluation--benchmark-results)
- [Interactive Web Application Architecture](#-interactive-web-application-architecture)
- [API Reference](#-api-reference)
- [Repository Structure](#-repository-structure)
- [Quickstart & Local Installation](#-quickstart--local-installation)
- [Production Deployment](#-production-deployment)
- [Citation & References](#-citation--references)

---

## ⚡ Key Highlights

- **167,147 Registered Typists**: Trained and evaluated on large-scale behavioral data from Aalto University (over 2.5 million sequences processed).
- **Sub-Millisecond Keystroke Telemetry**: Browser-based event capture capturing exact key dwell and flight latencies using `window.performance.now()`.
- **Dual-Task Biometric Capabilities**:
  - **1:1 Verification**: Siamese Twin LSTM projecting typing dynamics into a normalized 32D metric space; decides whether a subject is **Genuine** or an **Impostor** using Euclidean distance against an Equal Error Rate threshold (**EER = 23.36%, Accuracy = 76.68%**).
  - **1:N Identification**: Attention-based Transformer Classifier determining who among 300 registered users typed the sequence (**Top-1 Accuracy = 51.25%**, beating Bidirectional LSTM by **+15.03%**).
- **Zero-Latency Async Backend**: Built with **FastAPI** and **Uvicorn**, running batch inference with PyTorch models optimized for CPU/GPU execution.
- **Glassmorphic Cyber-Security UI**: Modern responsive dashboard with live telemetry meters, top-5 probability distribution charts, and interactive enrollment.

---

## ⏱️ Behavioral Biometrics & Feature Engineering

Typing behavior is a continuous behavioral biometric: the human neuromuscular timing of striking, holding, and moving between keys cannot easily be copied or forged. 

For every keystroke $i$, the engine extracts three fundamental temporal timing features in milliseconds:

$$\begin{aligned}
\mathbf{H}_i \quad &\text{(Hold / Dwell Time)} &&= T_{\text{release}}(i) - T_{\text{press}}(i) \\
\mathbf{DD}_i \quad &\text{(Down-Down / Digraph Latency)} &&= T_{\text{press}}(i) - T_{\text{press}}(i-1) \quad (\mathbf{DD}_0 = 0) \\
\mathbf{UD}_i \quad &\text{(Up-Down / Flight Time)} &&= T_{\text{press}}(i) - T_{\text{release}}(i-1) \quad (\mathbf{UD}_0 = 0)
\end{aligned}$$

```
Keystroke i-1: [ PRESS ]----------------[ RELEASE ]
                   |                        |
                   |<------- UD_i --------->|
                   |                        |
Keystroke i:       |       [ PRESS ]-----------------[ RELEASE ]
                   |<--- DD_i ----->|       |<--- H_i ------>|
```

Each sequence is normalized using a `StandardScaler` fitted exclusively on the training split flattened across time:

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

---

## 📊 Dataset Preprocessing & Augmentation

| Metric | Full Dataset (Siamese 1:1) | Subsampled Dataset (Identification 1:N) |
| :--- | :--- | :--- |
| **Source Data** | Aalto 136M Keystrokes (168,593 TSV files, 1.57 GB) | 300 Randomly sampled registered users |
| **Unique Users** | **167,147 users** (15 typed sentences each) | **300 users** (299 classes evaluated) |
| **Total Samples** | **2,507,205 sequences** | **78,474 augmented sequences** |
| **Window Strategy** | Fixed window: 1 sequence per sentence | Sliding window: `stride = 2`, `min_len = 5` |
| **Sequence Length ($T$)** | **50 keystrokes** (zero-padded if shorter) | **15 keystrokes** (zero-padded if shorter) |
| **Split Ratio** | 80% Train (`2,005,764`), 20% Test (`501,441`) | 80% Train (`62,779`), 20% Test (`15,695`) |
| **Input Shape** | `(Batch, 50, 3)` | `(Batch, 15, 3)` |

---

## 🧠 Deep Learning Architectures

```
                     ┌────────────────────────────────────────────────────────┐
                     │          Keystroke Timing Vector: [H, DD, UD]          │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [ Siamese LSTM (1:1) ]                                      [ Transformer Encoder (1:N) ]
  ┌──────────────────────────────┐                            ┌──────────────────────────────────┐
  │ Input: (B, 50, 3)            │                            │ Input: (B, 15, 3)                │
  │ Twin LSTM (Hidden: 128)      │                            │ Linear Projection (3 -> 64)      │
  │ Dense Embedding (128 -> 32)  │                            │ Sinusoidal Positional Encoding   │
  │ L2 Normalized Latent z       │                            │ 2x Transformer Layers (4 Heads)  │
  └──────────────┬───────────────┘                            │ Feed-Forward (128) + Dropout 0.1 │
                 │                                            │ Mean Temporal Pooling            │
                 ▼                                            │ Linear Classifier (64 -> 300)    │
  Pairwise Euclidean Distance:                                └────────────────┬─────────────────┘
  d = ||z1 - z2||_2                                                            │
  If d < 0.70 => Verified ✅                                                   ▼
  Else        => Impostor ❌                                   Softmax Distribution over 300 IDs
```

### 1. Siamese LSTM (1:1 Verification)
* **Goal**: Determine whether two typed sequences originate from the same user without needing to retrain on new users.
* **Architecture**:
  * Input: `(batch_size, 50, 3)`
  * Shared Recurrent Core: Single-layer `LSTM(input_size=3, hidden_size=128, batch_first=True)`
  * Projection Head: `nn.Linear(128, 32)` mapping hidden state $h_n$ to a compact 32-dimensional embedding space.
* **Loss Function**: Contrastive Loss with margin $m = 1.0$:
  $$\mathcal{L}(z_1, z_2, y) = y \cdot d(z_1, z_2)^2 + (1 - y) \cdot \max\big(0, m - d(z_1, z_2)\big)^2$$
  where $y=1$ for genuine pairs and $y=0$ for impostor pairs.

### 2. Transformer Encoder Classifier (1:N Identification)
* **Goal**: Multi-class classification across 300 candidate profiles.
* **Architecture**:
  * Linear Token Projection: $\mathbb{R}^3 \to \mathbb{R}^{64}$ (`d_model = 64`).
  * Sinusoidal Positional Encoding preserving keystroke sequence order:
    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
  * Multi-Head Self-Attention: 2 layers of `TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)`.
  * Global Temporal Average Pooling over length dimension.
  * Classification Head: `nn.Linear(64, 300)`.

### 3. Bidirectional LSTM Classifier (1:N Identification)
* **Goal**: Recurrent baseline comparison.
* **Architecture**:
  * Input: `(batch_size, 15, 3)`
  * Bidirectional Core: `nn.LSTM(input_size=3, hidden_size=128, batch_first=True, bidirectional=True)`
  * Regularization: `nn.Dropout(p=0.25)`
  * Output Projection: `nn.Linear(256, 300)` taking concatenated forward and backward final hidden states $[h_{\text{forward}}; h_{\text{backward}}]$.

---

## ⚙️ Training Hyperparameters & Experimental Setup

All models were trained in a dual-GPU Kaggle accelerator environment (`2x NVIDIA T4`, `DataParallel`, `PyTorch 2.x`).

| Parameter | Siamese LSTM (Verification) | Transformer Encoder (ID) | Bidirectional LSTM (ID) |
| :--- | :--- | :--- | :--- |
| **Task Scope** | 1:1 Binary Metric Verification | 1:N 300-Class Classification | 1:N 300-Class Classification |
| **Sequence Length ($T$)** | 50 keystrokes | 15 keystrokes | 15 keystrokes |
| **Batch Size** | 256 | 256 | 256 |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| **Learning Rate ($\eta$)** | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ |
| **Loss Function** | Contrastive Loss ($\text{margin}=1.0$) | CrossEntropyLoss | CrossEntropyLoss |
| **Epochs** | 30 | 60 | 60 |
| **Number of GPUs** | 2x NVIDIA T4 (`DataParallel`) | 2x NVIDIA T4 (`DataParallel`) | 2x NVIDIA T4 (`DataParallel`) |
| **Total Parameters** | ~72,000 | ~77,000 | ~170,000 |

---

## 📈 Evaluation & Benchmark Results

### 1. Model Performance Summary

| Model | Task | Test Metric | Score | Training Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| 🛡️ **Siamese LSTM** | 1:1 Verification | **Equal Error Rate (EER)** | **0.2336 (23.36%)** | — |
| 🛡️ **Siamese LSTM** | 1:1 Verification | **Accuracy @ EER Threshold** | **76.68%** | — |
| ⚡ **Transformer Encoder** | 1:N Identification | **Top-1 Accuracy** | **51.25%** | 61.50% |
| 🔄 **Bidirectional LSTM** | 1:N Identification | **Top-1 Accuracy** | **36.22%** | 65.53% |

### 2. Key Scientific Observations

1. **Transformer vs. Recurrent Efficiency**: The **Transformer Encoder achieved 51.25% Top-1 accuracy** over 300 classes, exceeding the Bidirectional LSTM (36.22%) by **+15.03%**. Self-attention directly models pairwise keystroke dependencies (e.g., digraph patterns between common letter combinations) regardless of their sequential distance.
2. **Siamese Generalization**: The Siamese model successfully learned a universal behavioral distance metric across **167k typists** that generalizes to unseen typists without fine-tuning, reaching an **EER of 23.36%**.

---

## 🖥️ Interactive Web Application Architecture

The system features a live browser application that captures raw keyboard hardware events:

1. **Millisecond Precision Capture**: Event listeners on `keydown` and `keyup` calculate exact timestamps using `performance.now()`.
2. **Real-Time Dynamic Telemetry**:
   - **Hold Duration Meter**: Visual gauge showing instantaneous finger-on-key duration.
   - **Flight Velocity**: Inter-key latency visualizer.
   - **Sequence Buffer**: Live counter tracking completion of the required sequence length ($T=50$ or $T=15$).
3. **Four Operating Modes**:
   - **1:1 Live Verification**: Verify live typing against benchmark users (`User 367392`, `User 225417`) or custom enrolled profiles with Euclidean distance progress gauges.
   - **1:N Identification**: Feed 15 keystrokes to both Transformer and BiLSTM simultaneously, displaying Top-5 ranked candidate predictions with softmax probabilities.
   - **Live Enrollment**: Enroll a user's unique typing profile directly through the browser.
   - **Architecture & Benchmarks**: Interactive structural diagrams and metric inspection.

---

## 🔌 API Reference

### 1. `POST /api/verify`
Runs Siamese 1:1 biometric comparison between two keystroke sequences.

**Payload**:
```json
{
  "reference_sequence": [[85.2, 120.4, 35.2], ...],  // 50 x 3 [H, DD, UD]
  "sample_sequence": [[90.1, 115.0, 24.9], ...]       // 50 x 3 [H, DD, UD]
}
```

**Response**:
```json
{
  "distance": 0.5421,
  "threshold": 0.70,
  "verified": true,
  "confidence_percent": 88.4,
  "verdict": "GENUINE_USER"
}
```

---

### 2. `POST /api/identify`
Executes 1:N identification across 300 registered profiles using both Transformer and BiLSTM.

**Payload**:
```json
{
  "sequence": [[80.0, 150.0, 70.0], ...]  // 15 x 3 [H, DD, UD]
}
```

**Response**:
```json
{
  "transformer_prediction": "367392",
  "transformer_top5": [
    {"user_id": "367392", "confidence": 0.724},
    {"user_id": "225417", "confidence": 0.115},
    {"user_id": "104928", "confidence": 0.042},
    {"user_id": "491823", "confidence": 0.031},
    {"user_id": "184920", "confidence": 0.022}
  ],
  "lstm_prediction": "367392",
  "lstm_top5": [...],
  "total_classes": 300
}
```

---

### 3. `GET /api/benchmark_samples`
Retrieves pre-computed test sequences from verified benchmark users (`367392` and `225417`).

### 4. `GET /api/health`
Health check returning server status, PyTorch version, active compute device (`cpu` / `cuda`), and loaded model parameters.

---

## 📂 Repository Structure

```
├── app.py                                # FastAPI backend & inference engine
├── requirements.txt                      # Optimized dependencies (CPU PyTorch)
├── Dockerfile                            # Production container specification
├── .gitignore                            # Excludes checkpoints, pycache & temp archives
├── keystroke-dynamics-authentication.ipynb# Complete Kaggle training & evaluation notebook
├── keystroke_models/                     # Trained models & preprocessing artifacts (~5.5MB)
│   ├── siamese_model.pt                  # Siamese LSTM weights (Verification)
│   ├── transformer_model.pt              # Transformer Encoder weights (Identification)
│   ├── lstm_model.pt                     # Bidirectional LSTM weights (Identification)
│   ├── scaler_full.pkl                   # StandardScaler for 50-step Siamese sequences
│   ├── scaler_id.pkl                     # StandardScaler for 15-step ID sequences
│   ├── label_encoder_id.pkl              # 300-Class identification label encoder
│   ├── label_encoder_full.pkl            # Full dataset user encoder (167k users)
│   └── demo_reference_samples.pkl        # Reference samples for benchmark users
├── static/                               # Frontend presentation assets
│   ├── css/
│   │   └── style.css                     # Glassmorphic cyberpunk styling
│   └── js/
│       └── app.js                        # Event telemetry, visual gauges & API orchestration
└── templates/
    └── index.html                        # Single-page application shell
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python `3.10` or `3.11`
- `git`

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/keystroke-dynamics-authentication.git
cd keystroke-dynamics-authentication
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python app.py
```
Open **[http://localhost:10000](http://localhost:10000)** (or the port displayed in your terminal) in your browser.

---

## 🐳 Production Deployment

### Option A: Deploy to Render (Recommended Free Cloud Hosting)

1. Fork or push this repository to your GitHub account.
2. Sign in to **[Render.com](https://render.com)**.
3. Click **New +** ➔ **Web Service** ➔ Select your repository.
4. Set the following configuration:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
5. Click **Deploy Web Service**.

### Option B: Docker Container

Build and run with Docker:

```bash
# Build Docker image
docker build -t keystroke-dynamics:latest .

# Run container on port 7860
docker run -p 7860:7860 keystroke-dynamics:latest
```
Access the application at `http://localhost:7860`.

---

## 📚 Citation & References

- **Dataset**: Dhakal, V., Feit, A. M., Kristensson, P. O., & Oulasvirta, A. (2018). *Observations on Typing from 136 Million Keystrokes*. ACM CHI Conference on Human Factors in Computing Systems. [https://userinterfaces.aalto.fi/136Mkeystrokes/](https://userinterfaces.aalto.fi/136Mkeystrokes/)
- **Siamese Networks**: Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1993). *Signature verification using a "Siamese" time delay neural network*. NeurIPS.
- **Attention & Transformers**: Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

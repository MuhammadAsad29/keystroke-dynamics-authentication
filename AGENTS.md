# Environment & System Knowledge (Persistent Memory)

> [!NOTE]
> This file is permanently memorized for this workspace. Every new chat session automatically reads this file first so inspection steps (Python location, package availability, model architecture, dataset info) are never repeated.

---

## 1. Python Environment & Execution Path

* **Python Executable**:
  ```powershell
  C:\Users\muham\AppData\Local\Programs\Python\Python311\python.exe
  ```
* **Python Version**: `3.11.x`
* **Installed Packages**:
  * `torch` (PyTorch with CUDA / CPU acceleration)
  * `torchvision`
  * `scikit-learn` (`sklearn`)
  * `numpy`
  * `scipy`
  * `streamlit`
  * `pillow` (`PIL`)
* **Shell & Execution Notes**:
  * Operating System: **Windows**
  * Shell: **PowerShell**
  * When running python commands or scripts, always use the explicit full path:
    ```powershell
    & "C:\Users\muham\AppData\Local\Programs\Python\Python311\python.exe" script.py
    ```
  * For local web servers (Flask, FastAPI, Streamlit), run with `BypassSandbox: true` so the host network and Python directory are directly accessible.

---

## 2. Project Architecture & Saved Model Checkpoints

This workspace implements **Keystroke Dynamics Behavioral Biometrics** (Aalto University 136M Keystroke Dataset).

### Directory: `keystroke_models/`
1. **Siamese LSTM (`siamese_model.pt`)**:
   * **Task**: 1:1 Biometric Verification (Authentication: Genuine vs. Impostor).
   * **Architecture**: `SiameseLSTM(input_size=3, hidden=128, embed_dim=32)`
   * **Sequence Length**: `SEQ_LEN = 50`
   * **Normalization**: Scaled with `scaler_full.pkl` (`StandardScaler` over `[hold_time, dd_time, ud_time]`).
   * **Evaluation Metric**: Equal Error Rate (EER) = `0.2336`, Accuracy @ EER = `76.68%`.
   * **Distance Metric**: Euclidean pairwise distance $d = \|\mathbf{z}_1 - \mathbf{z}_2\|_2$.
   * **Decision**: If $d < \tau$ (default $\tau \approx 0.70 - 0.85$), **"Verified ✅"**; else **"Impostor ❌"**.

2. **Bidirectional LSTM Classifier (`lstm_model.pt`)**:
   * **Task**: 1:N User Identification (Classify among 300 registered users).
   * **Architecture**: `LSTMClassifier(input_size=3, hidden=128, bidirectional=True, num_classes=300)`
   * **Sequence Length**: `ID_SEQ_LEN = 15`
   * **Normalization**: Scaled with `scaler_id.pkl`.
   * **Top-1 Accuracy**: `36.22%`.

3. **Transformer Encoder Classifier (`transformer_model.pt`)**:
   * **Task**: 1:N User Identification.
   * **Architecture**: `TransformerClassifier(input_size=3, d_model=64, nhead=4, num_layers=2, num_classes=300)` with `PositionalEncoding(d_model=64, max_len=15)`.
   * **Sequence Length**: `ID_SEQ_LEN = 15`.
   * **Normalization**: Scaled with `scaler_id.pkl`.
   * **Top-1 Accuracy**: `51.25%`.

4. **Pickled Artifacts**:
   * `scaler_full.pkl`: `StandardScaler` for Siamese (50-timestep sequences).
   * `scaler_id.pkl`: `StandardScaler` for Identification (15-timestep sequences).
   * `label_encoder_id.pkl`: Label encoder for 300 identification classes.
   * `label_encoder_full.pkl`: Full dataset user encoder (167k users).
   * `demo_reference_samples.pkl`: Reference evaluation samples for pre-loaded benchmark users (`"367392"`, `"225417"`).

---

## 3. Feature Extraction Pipeline
For each keystroke event $i$:
* **$H_i$ (Hold / Dwell Time)**: $T_{\text{release}}(i) - T_{\text{press}}(i)$
* **$DD_i$ (Down-Down / Digraph Latency)**: $T_{\text{press}}(i) - T_{\text{press}}(i-1)$ (with $DD_0 = 0$)
* **$UD_i$ (Up-Down / Flight Time)**: $T_{\text{press}}(i) - T_{\text{release}}(i-1)$ (with $UD_0 = 0$)
* Timing units: **Milliseconds (ms)**, directly matching browser `performance.now()`.

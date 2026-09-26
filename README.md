# WeatherBlend AI: Context-Aware Forecast Blending Engine (SIH26081 MVP)

> **Hackathon Prototype for SIH26081:** A context-aware weather forecast blending engine that dynamically learns which forecast stream (NWP, Multi-Model Ensemble, or AI-Model) to trust based on spatial region, season, lead horizon, and prevailing weather regime—replacing naive arithmetic averaging.

---

## 🚀 Quick Start (One-Command Demo Execution)

Run the full end-to-end pipeline (Data Generation → Baselines → Stage 2 Stacking → Stage 3 Gating → Interactive Frontend Dashboard) with a single command:

### On Windows:
```cmd
run_demo.bat
```

### On Linux / macOS / Bash:
```bash
chmod +x run_demo.sh
./run_demo.sh
```

### Or using `make`:
```bash
make demo
```

---

## ⚡ Running Frontend & Backend Separately

### 1. Launch FastAPI Backend Service
```bash
python run_backend.py
# or
make backend
```
Access interactive API documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Launch Streamlit Frontend Application
```bash
python run_frontend.py
# or
make frontend
```
Access interactive web dashboard at: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Architecture & Directory Structure

```
weather-blend-mvp/
├── frontend/                     # Frontend Web Dashboard App
│   ├── app.py                    # Streamlit interactive application
│   └── requirements.txt          # Frontend dependencies
├── backend/                      # Backend Service & ML Engine
│   ├── main.py                   # FastAPI application server (REST API)
│   ├── requirements.txt          # Backend dependencies
│   ├── src/                      # Machine learning engine source code
│   │   ├── __init__.py
│   │   ├── generate_data.py      # Synthetic dataset generator
│   │   ├── baselines.py          # Stage 1 baseline models
│   │   ├── stacking_model.py     # Stage 2 LightGBM stacking blender
│   │   └── gating_model.py       # Stage 3 Adaptive Softmax Gating & Two-Stage Head
│   ├── models/                   # Model artifacts (.joblib)
│   │   ├── gating_model.joblib
│   │   └── stacking_model.joblib
│   └── data/                     # Dataset & generated results (.parquet & .csv)
├── run_backend.py                # Python runner for FastAPI backend
├── run_frontend.py               # Python runner for Streamlit frontend
├── requirements.txt              # Root master requirements
├── run_demo.bat                  # One-command runner for Windows
├── run_demo.sh                   # One-command runner for Bash/Linux
└── Makefile                      # Make targets
```

---

## 🏗️ Model Pipeline Progression

```
[ NWP Model (Source 1) ] ---\
[ Ensemble (Source 2) ] ----+--> [ Context Feature Engine ] --> [ Stage 3 Adaptive Softmax Gate ] --> Blended Temperature Forecast
[ AI-Model (Source 3) ] ---/   (Diffs, Spread, Regimes,        [ Two-Stage Occurrence + Tweedie ] --> Blended Rainfall Forecast
                                Causal 7d Errors)
```

### 1. **Stage 1 Baselines (`backend/src/baselines.py`)**
* **Equal-Weight Ensemble Average:** Standard mean $\frac{1}{3}(S_1 + S_2 + S_3)$.
* **Best Single Historical Model:** Picks source with lowest training MAE per `(variable, lead_hours, season)` group.
* **Inverse-Error Skill Weighting:** Dynamic weights $w_m \propto (E_m + \epsilon)^{-2}$ based on training MAE $E_m$.
* **Bias-Corrected Model Average:** Subtracts additive training bias per source, then averages.

### 2. **Stage 2 Stacking Blender (`backend/src/stacking_model.py`)**
* **LightGBM Regressor** with residual target learning ($\delta = \text{observation} - \text{ens\_mean}$).
* Features: Raw sources, pairwise differences, ensemble mean & spread, lead horizon, trigonometric seasonal encodings ($\sin/\cos$), spatial grid coordinates, weather regime clusters, and 7-day causal rolling historical error.

### 3. **Stage 3 Adaptive Gating & Two-Stage Rainfall Head (`backend/src/gating_model.py`)**
* **Temperature Head:** Multi-head GBDT predicting dynamic **Softmax weights** ($w_1, w_2, w_3 \ge 0$, $\sum w_i = 1$) over the 3 sources: $\hat{y}_{\text{temp}} = \sum w_m S_m + b_{\text{bias}}$.
* **Rainfall Two-Stage Head:**
  * **Stage 1 (Occurrence Classifier):** `LGBMClassifier` predicting probability of precipitation $P(\text{rain} > 0.1\text{ mm})$. Evaluated via **Brier Score** and **CSI Threat Score**.
  * **Stage 2 (Conditional Regressor):** `LGBMRegressor` trained with **Tweedie loss** ($\text{power}=1.5$) on positive rain occurrences.
  * **Combined Forecast:** $\hat{y}_{\text{rain}} = P(\text{rain} > 0) \times \hat{A}_{\text{rain}}$, with drizzle noise ($P < 0.15$) truncated to 0.

---

## 📊 Evaluation Results (Year 2 Test Set)

Strict time-based evaluation (Train = 2023 / Year 1, Test = 2024 / Year 2; 73,000 test samples).

### **Rainfall Metrics (Continuous & Probabilistic)**
| Model / Strategy | MAE (mm) | RMSE (mm) | Skill Score (MAE) | Brier Score | CSI (Threat Score) |
|---|---|---|---|---|---|
| Persistence Baseline | 10.7847 | 22.7628 | 0.0000 | 0.2552 | 0.3778 |
| Source 1 (NWP Physics Model) | 5.8967 | 10.3669 | 0.4532 | 0.1846 | 0.5143 |
| Source 2 (Multi-Model Ensemble) | 2.3345 | 3.7957 | 0.7835 | 0.0944 | 0.6656 |
| Source 3 (AI Neural Forecast) | 2.0550 | 3.2536 | 0.8094 | 0.0790 | 0.7151 |
| 1. Equal-Weight Average | 2.8656 | 4.2204 | 0.7343 | 0.0895 | 0.6824 |
| 2. Best Single Historical | 1.9316 | 3.0864 | 0.8209 | 0.0773 | 0.7234 |
| 3. Inverse-Error Weighting | 1.9337 | 2.6477 | 0.8207 | 0.0690 | 0.7658 |
| 4. Bias-Corrected Average | 2.2442 | 3.5534 | 0.7919 | 0.0821 | 0.7144 |
| 5. LightGBM Blender (Stage 2) | 1.0546 | 1.8994 | 0.9022 | 0.0744 | 0.7600 |
| **6. Adaptive Gating Head (Stage 3)** | **0.9846** | **2.2091** | **0.9087** | **0.0529** | **0.8049** |

---

### **Temperature Metrics (°C)**
| Model / Strategy | MAE (°C) | RMSE (°C) | Skill Score (RMSE) | Skill Score (MAE) |
|---|---|---|---|---|
| Persistence Baseline | 1.1275 | 1.4090 | 0.0000 | 0.0000 |
| Source 1 (NWP Physics Model) | 2.0094 | 3.0725 | -1.1805 | -0.7822 |
| Source 2 (Multi-Model Ensemble) | 1.3929 | 1.8105 | -0.2849 | -0.2353 |
| Source 3 (AI Neural Forecast) | 1.1796 | 1.4921 | -0.0589 | -0.0462 |
| 1. Equal-Weight Average | 0.9361 | 1.2880 | 0.0859 | 0.1698 |
| 2. Best Single Historical | 1.0266 | 1.3572 | 0.0368 | 0.0895 |
| 3. Inverse-Error Weighting | 0.7457 | 0.9849 | 0.3010 | 0.3387 |
| 4. Bias-Corrected Average | 0.9313 | 1.2792 | 0.0922 | 0.1740 |
| 5. LightGBM Blender (Stage 2) | 0.8395 | 1.0658 | 0.2436 | 0.2555 |
| **6. Adaptive Gating Head (Stage 3)** | **0.8361** | **1.1221** | **0.2037** | **0.2585** |

"""
Stage 3: Adaptive Gating Model & Two-Stage Rainfall Head for Weather Blend MVP (SIH26081)

Architecture:
1. Temperature Head: Softmax Gating Model
   - Trains multi-head GBDT predicting dynamic softmax weights over 3 sources.
   - Forecast = sum(w_m * S_m) + bias_correction.

2. Rainfall Head: Two-Stage Occurrence + Conditional Tweedie Head
   - Stage 1: LGBMClassifier predicting P(rain > 0) with Binary Cross-Entropy loss.
   - Stage 2: LGBMRegressor predicting conditional rain amount using Tweedie loss.
   - Combined Forecast = P(rain > 0) * Conditional_Amount.

Evaluates MAE, RMSE, Skill Score, Brier Score, and Critical Success Index (CSI).
Breakdown by lead_time and weather_regime.
Outputs final grand comparison table across all stages (Baselines, Stage 2, Stage 3).
Saves model artifact to /models/gating_model.joblib.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import softmax
from typing import Dict, List, Tuple

from src.baselines import (
    calculate_mae,
    calculate_rmse,
    calculate_skill_score,
    add_persistence_column,
    EqualWeightModel,
    BestSingleHistoricalModel,
    InverseErrorWeightingModel,
    BiasCorrectedAverageModel
)
from src.stacking_model import engineer_stacking_features, LightGBMStackingBlender, FEATURE_COLS

# -----------------------------------------------------------------------------
# PROBABILISTIC & CATEGORICAL METRICS
# -----------------------------------------------------------------------------

def calculate_brier_score(y_true_bin: np.ndarray, p_pred: np.ndarray) -> float:
    """
    Brier Score for probability of precipitation.
    Brier = mean((p_pred - y_true_bin)^2). Range: 0.0 (perfect) to 1.0.
    """
    return float(np.mean((p_pred - y_true_bin) ** 2))

def calculate_csi(
    y_true_rain: np.ndarray,
    p_pred: np.ndarray,
    rain_threshold: float = 0.1,
    prob_threshold: float = 0.35
) -> float:
    """
    Critical Success Index (CSI / Threat Score) for rainfall occurrence.
    CSI = Hits / (Hits + False_Alarms + Misses)
    Range: 0.0 to 1.0 (perfect).
    """
    true_occ = y_true_rain > rain_threshold
    pred_occ = p_pred > prob_threshold

    hits = np.sum(pred_occ & true_occ)
    false_alarms = np.sum(pred_occ & ~true_occ)
    misses = np.sum(~pred_occ & true_occ)

    denom = hits + false_alarms + misses
    if denom == 0:
        return 1.0
    return float(hits / denom)

# -----------------------------------------------------------------------------
# STAGE 3 ADAPTIVE GATING & TWO-STAGE MODEL CLASS
# -----------------------------------------------------------------------------

class AdaptiveGatingBlender:
    """
    Stage 3 Context-Aware Adaptive Gating & Two-Stage Rainfall Model.
    """
    def __init__(self, tau: float = 0.8):
        self.name = "6. Adaptive Gating Head (Stage 3)"
        self.tau = tau
        self.feature_cols = FEATURE_COLS
        
        # Temperature Softmax Gating Regressors (1 per source)
        self.temp_gating_models: Dict[str, lgb.LGBMRegressor] = {}
        self.temp_bias: float = 0.0
        
        # Two-Stage Rainfall Head
        self.rain_classifier: lgb.LGBMClassifier = None
        self.rain_regressor: lgb.LGBMRegressor = None

    def fit(self, df_train: pd.DataFrame):
        # ---------------------------------------------------------------------
        # 1. TEMPERATURE ADAPTIVE SOFTMAX GATING HEAD
        # ---------------------------------------------------------------------
        sub_t = df_train[df_train["variable"] == "temperature"].copy()
        X_t = sub_t[self.feature_cols]
        y_t = sub_t["observation"].values
        
        s1_t = sub_t["source_1"].values
        s2_t = sub_t["source_2"].values
        s3_t = sub_t["source_3"].values
        
        # Compute target softmax weights based on absolute forecast errors
        errs_t = np.column_stack([
            np.abs(s1_t - y_t),
            np.abs(s2_t - y_t),
            np.abs(s3_t - y_t)
        ])
        target_weights_t = softmax(-errs_t / self.tau, axis=1) # (N, 3)

        # Fit 3 GBDTs to predict target weights
        for idx, src in enumerate(["source_1", "source_2", "source_3"]):
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=25,
                random_state=42,
                verbosity=-1
            )
            model.fit(X_t, target_weights_t[:, idx])
            self.temp_gating_models[src] = model

        # Learn global residual bias on train set for temperature
        w_raw = np.column_stack([self.temp_gating_models[src].predict(X_t) for src in ["source_1", "source_2", "source_3"]])
        w_norm = softmax(w_raw, axis=1)
        blended_t_train = np.sum(w_norm * np.column_stack([s1_t, s2_t, s3_t]), axis=1)
        self.temp_bias = float(np.mean(y_t - blended_t_train))

        # ---------------------------------------------------------------------
        # 2. TWO-STAGE RAINFALL HEAD
        # ---------------------------------------------------------------------
        sub_r = df_train[df_train["variable"] == "rainfall"].copy()
        X_r = sub_r[self.feature_cols]
        y_r = sub_r["observation"].values
        y_r_bin = (y_r > 0.1).astype(int)

        # Stage 1 Classifier: P(rain > 0)
        self.rain_classifier = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=25,
            random_state=42,
            verbosity=-1
        )
        self.rain_classifier.fit(X_r, y_r_bin)

        # Stage 2 Conditional Regressor (Tweedie loss on non-zero rain events)
        rain_mask = y_r > 0.1
        X_r_pos = X_r[rain_mask]
        y_r_pos = y_r[rain_mask]

        self.rain_regressor = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=350,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=25,
            random_state=42,
            verbosity=-1
        )
        self.rain_regressor.fit(X_r_pos, y_r_pos)

    def predict_temperature(self, df_temp: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (blended_forecast, softmax_weights)."""
        X = df_temp[self.feature_cols]
        s1 = df_temp["source_1"].values
        s2 = df_temp["source_2"].values
        s3 = df_temp["source_3"].values
        sources = np.column_stack([s1, s2, s3])

        logits = np.column_stack([self.temp_gating_models[src].predict(X) for src in ["source_1", "source_2", "source_3"]])
        weights = softmax(logits, axis=1)

        preds = np.sum(weights * sources, axis=1) + self.temp_bias
        return preds, weights

    def predict_rainfall(self, df_rain: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (blended_forecast, prob_rain)."""
        X = df_rain[self.feature_cols]

        # Stage 1: P(rain > 0)
        probs_rain = self.rain_classifier.predict_proba(X)[:, 1]

        # Stage 2: Conditional rain volume
        cond_amount = self.rain_regressor.predict(X)
        cond_amount = np.maximum(0.0, cond_amount)

        # Combined expected value forecast
        preds = probs_rain * cond_amount
        
        # Truncate low-probability drizzle noise (< 15% rain chance)
        preds = np.where(probs_rain < 0.15, 0.0, preds)
        return preds, probs_rain

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df_test))
        
        # Temperature
        mask_t = df_test["variable"].values == "temperature"
        if np.any(mask_t):
            sub_t = df_test[mask_t]
            t_preds, _ = self.predict_temperature(sub_t)
            preds[mask_t] = t_preds

        # Rainfall
        mask_r = df_test["variable"].values == "rainfall"
        if np.any(mask_r):
            sub_r = df_test[mask_r]
            r_preds, _ = self.predict_rainfall(sub_r)
            preds[mask_r] = r_preds

        return preds

# -----------------------------------------------------------------------------
# GRAND COMPARATIVE EVALUATION SUITE
# -----------------------------------------------------------------------------

def evaluate_grand_comparison(
    df_train_feat: pd.DataFrame,
    df_test_feat: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, AdaptiveGatingBlender]:
    """
    Fits all baseline models, Stage 2 LightGBM, and Stage 3 Gating Head.
    Generates unified comparative tables across all stages.
    """
    df_test_feat = add_persistence_column(df_test_feat)
    y_true = df_test_feat["observation"].values
    y_pers = df_test_feat["persistence"].values

    # Train baselines
    m_eq = EqualWeightModel()
    m_best = BestSingleHistoricalModel()
    m_inv = InverseErrorWeightingModel()
    m_bias = BiasCorrectedAverageModel()

    m_best.fit(df_train_feat)
    m_inv.fit(df_train_feat)
    m_bias.fit(df_train_feat)

    # Train Stage 2 LightGBM
    m_stage2 = LightGBMStackingBlender()
    print("Training Stage 2 LightGBM Blender...")
    m_stage2.fit(df_train_feat)

    # Train Stage 3 Adaptive Gating & Two-Stage Model
    m_stage3 = AdaptiveGatingBlender()
    print("Training Stage 3 Adaptive Gating & Two-Stage Rainfall Head...")
    m_stage3.fit(df_train_feat)

    # Dictionary of all 9 models
    all_models = {
        "Source 1 (NWP)": df_test_feat["source_1"].values,
        "Source 2 (Ensemble)": df_test_feat["source_2"].values,
        "Source 3 (AI-Model)": df_test_feat["source_3"].values,
        "Persistence": y_pers,
        "1. Equal-Weight Avg": m_eq.predict(df_test_feat),
        "2. Best Single Historical": m_best.predict(df_test_feat),
        "3. Inverse-Error Weighting": m_inv.predict(df_test_feat),
        "4. Bias-Corrected Avg": m_bias.predict(df_test_feat),
        "5. LightGBM Blender (Stage 2)": m_stage2.predict(df_test_feat),
        "6. Adaptive Gating Head (Stage 3)": m_stage3.predict(df_test_feat)
    }

    # Extract Stage 3 Probabilities for Rainfall evaluation
    sub_r_test = df_test_feat[df_test_feat["variable"] == "rainfall"].copy()
    _, stage3_p_rain = m_stage3.predict_rainfall(sub_r_test)
    
    # -------------------------------------------------------------------------
    # 1. OVERALL COMPARISON TABLE
    # -------------------------------------------------------------------------
    overall_rows = []
    for var in ["temperature", "rainfall"]:
        mask = df_test_feat["variable"].values == var
        sub_true = y_true[mask]
        sub_pers = y_pers[mask]
        ref_rmse = calculate_rmse(sub_true, sub_pers)
        ref_mae = calculate_mae(sub_true, sub_pers)

        for name, preds in all_models.items():
            sub_pred = preds[mask]
            mae = calculate_mae(sub_true, sub_pred)
            rmse = calculate_rmse(sub_true, sub_pred)
            ss_rmse = calculate_skill_score(rmse, ref_rmse)
            ss_mae = calculate_skill_score(mae, ref_mae)

            # Rainfall specific metrics (Brier Score & CSI)
            brier, csi = np.nan, np.nan
            if var == "rainfall":
                y_bin = (sub_true > 0.1).astype(int)
                if name == "6. Adaptive Gating Head (Stage 3)":
                    p_vec = stage3_p_rain
                else:
                    # Proxy probability scaling for deterministic baselines
                    p_vec = np.clip(sub_pred / 15.0, 0.0, 1.0)
                
                brier = calculate_brier_score(y_bin, p_vec)
                csi = calculate_csi(sub_true, p_vec, prob_threshold=0.35)

            overall_rows.append({
                "variable": var,
                "model": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "Skill_Score_RMSE": round(ss_rmse, 4),
                "Skill_Score_MAE": round(ss_mae, 4),
                "Brier_Score": round(brier, 4) if not np.isnan(brier) else "N/A",
                "CSI": round(csi, 4) if not np.isnan(csi) else "N/A"
            })

    df_overall = pd.DataFrame(overall_rows)

    # -------------------------------------------------------------------------
    # 2. LEAD TIME BREAKDOWN TABLE
    # -------------------------------------------------------------------------
    lead_rows = []
    for var in ["temperature", "rainfall"]:
        for lead_h in sorted(df_test_feat["lead_hours"].unique()):
            mask = (df_test_feat["variable"].values == var) & (df_test_feat["lead_hours"].values == lead_h)
            sub_true = y_true[mask]
            sub_pers = y_pers[mask]
            ref_rmse = calculate_rmse(sub_true, sub_pers)

            for name, preds in all_models.items():
                sub_pred = preds[mask]
                mae = calculate_mae(sub_true, sub_pred)
                rmse = calculate_rmse(sub_true, sub_pred)
                ss_rmse = calculate_skill_score(rmse, ref_rmse)

                brier, csi = np.nan, np.nan
                if var == "rainfall":
                    y_bin = (sub_true > 0.1).astype(int)
                    if name == "6. Adaptive Gating Head (Stage 3)":
                        sub_mask_r = (sub_r_test["lead_hours"].values == lead_h)
                        p_vec = stage3_p_rain[sub_mask_r]
                    else:
                        p_vec = np.clip(sub_pred / 15.0, 0.0, 1.0)
                    brier = calculate_brier_score(y_bin, p_vec)
                    csi = calculate_csi(sub_true, p_vec)

                lead_rows.append({
                    "variable": var,
                    "lead_hours": lead_h,
                    "model": name,
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                    "Skill_Score_RMSE": round(ss_rmse, 4),
                    "Brier_Score": round(brier, 4) if not np.isnan(brier) else "N/A",
                    "CSI": round(csi, 4) if not np.isnan(csi) else "N/A"
                })
    df_lead = pd.DataFrame(lead_rows)

    # -------------------------------------------------------------------------
    # 3. WEATHER REGIME BREAKDOWN TABLE
    # -------------------------------------------------------------------------
    regime_rows = []
    regimes = sorted(df_test_feat["weather_regime"].unique())
    for var in ["temperature", "rainfall"]:
        for regime in regimes:
            mask = (df_test_feat["variable"].values == var) & (df_test_feat["weather_regime"].values == regime)
            if not np.any(mask):
                continue
            sub_true = y_true[mask]
            sub_pers = y_pers[mask]
            ref_rmse = calculate_rmse(sub_true, sub_pers)

            for name, preds in all_models.items():
                sub_pred = preds[mask]
                mae = calculate_mae(sub_true, sub_pred)
                rmse = calculate_rmse(sub_true, sub_pred)
                ss_rmse = calculate_skill_score(rmse, ref_rmse)

                brier, csi = np.nan, np.nan
                if var == "rainfall":
                    y_bin = (sub_true > 0.1).astype(int)
                    if name == "6. Adaptive Gating Head (Stage 3)":
                        sub_mask_reg = (sub_r_test["weather_regime"].values == regime)
                        p_vec = stage3_p_rain[sub_mask_reg]
                    else:
                        p_vec = np.clip(sub_pred / 15.0, 0.0, 1.0)
                    brier = calculate_brier_score(y_bin, p_vec)
                    csi = calculate_csi(sub_true, p_vec)

                regime_rows.append({
                    "variable": var,
                    "weather_regime": regime,
                    "model": name,
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                    "Skill_Score_RMSE": round(ss_rmse, 4),
                    "Brier_Score": round(brier, 4) if not np.isnan(brier) else "N/A",
                    "CSI": round(csi, 4) if not np.isnan(csi) else "N/A"
                })
    df_regime = pd.DataFrame(regime_rows)

    return df_overall, df_lead, df_regime, m_stage3


def main():
    print("Loading synthetic weather dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "weather_blend_bihar_2year.parquet")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")

    df = pd.read_parquet(data_path)

    print("Engineering contextual features...")
    df_feat = engineer_stacking_features(df)

    # STRICT TIME-BASED TRAIN/TEST SPLIT
    # Train: Year 1 (2023) | Test: Year 2 (2024)
    train_mask = df_feat["init_time"] < "2024-01-01"
    df_train = df_feat[train_mask].copy()
    df_test = df_feat[~train_mask].copy()

    print(f"\nTime-Based Train/Test Split:")
    print(f"  - Train Set (2023): {len(df_train):,} rows")
    print(f"  - Test Set  (2024): {len(df_test):,} rows")

    # Fit and evaluate Stage 3 along with all previous stages
    df_overall, df_lead, df_regime, stage3_model = evaluate_grand_comparison(df_train, df_test)

    # Save trained Stage 3 model artifact
    model_save_path = os.path.join(models_dir, "gating_model.joblib")
    print(f"\nSaving trained Stage 3 model to: {model_save_path}")
    joblib.dump(stage3_model, model_save_path)

    # Save CSV evaluation results
    output_overall_csv = os.path.join(PROJECT_ROOT, "data", "stage3_grand_comparison.csv")
    output_lead_csv = os.path.join(PROJECT_ROOT, "data", "stage3_lead_time_comparison.csv")
    output_regime_csv = os.path.join(PROJECT_ROOT, "data", "stage3_regime_comparison.csv")

    df_overall.to_csv(output_overall_csv, index=False)
    df_lead.to_csv(output_lead_csv, index=False)
    df_regime.to_csv(output_regime_csv, index=False)

    # PRINT GRAND COMPARISON TABLES
    print("\n" + "=" * 95)
    print("STAGE 3 ADAPTIVE GATING & TWO-STAGE HEAD vs PREVIOUS BASELINES (YEAR 2 TEST SET)")
    print("=" * 95)

    for var in ["temperature", "rainfall"]:
        print(f"\n>>> GRAND COMPARISON TABLE - VARIABLE: {var.upper()}")
        print("-" * 95)
        sub = df_overall[df_overall["variable"] == var][
            ["model", "MAE", "RMSE", "Skill_Score_RMSE", "Skill_Score_MAE", "Brier_Score", "CSI"]
        ]
        print(sub.to_string(index=False))

    print("\n" + "=" * 95)
    print("MAE BY LEAD TIME (TEMPERATURE °C)")
    print("=" * 95)
    pivot_temp_lead = df_lead[df_lead["variable"] == "temperature"].pivot(index="model", columns="lead_hours", values="MAE")
    print(pivot_temp_lead.round(3).to_string())

    print("\n" + "=" * 95)
    print("RAINFALL MAE BY WEATHER REGIME (mm)")
    print("=" * 95)
    pivot_rain_regime = df_regime[df_regime["variable"] == "rainfall"].pivot(index="model", columns="weather_regime", values="MAE")
    print(pivot_rain_regime.round(3).to_string())

    print("\n" + "=" * 95)
    print("RAINFALL PROBABILISTIC SKILL: BRIER SCORE & CSI BY WEATHER REGIME (STAGE 3)")
    print("=" * 95)
    stage3_reg_sub = df_regime[(df_regime["variable"] == "rainfall") & (df_regime["model"] == "6. Adaptive Gating Head (Stage 3)")]
    print(stage3_reg_sub[["weather_regime", "MAE", "RMSE", "Brier_Score", "CSI"]].to_string(index=False))

    print(f"\nStage 3 execution complete. Saved artifacts:")
    print(f"  - Model Artifact: {model_save_path}")
    print(f"  - Grand Comparison: {output_overall_csv}")
    print(f"  - Lead Time Comparison: {output_lead_csv}")
    print(f"  - Regime Comparison: {output_regime_csv}")

if __name__ == "__main__":
    main()

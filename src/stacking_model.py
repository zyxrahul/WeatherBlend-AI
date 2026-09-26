"""
Stage 2: Stacking / Machine Learning Blending Model for Weather Blend MVP (SIH26081)

Trains LightGBM regressors to dynamically blend 3 forecast sources based on context.
Uses residual learning: predicts delta = (observation - ensemble_mean).

Features:
- Forecast sources: source_1, source_2, source_3
- Pairwise forecast differences: (source_1 - source_2), (source_1 - source_3), (source_2 - source_3)
- Ensemble stats: ensemble mean (ens_mean), ensemble spread (ens_spread)
- Lead time: lead_hours (24h, 48h, 72h, 120h)
- Season encodings: sin/cos of month and day of year
- Spatial coordinates: lat, lon
- Weather regime: weather_regime categorical label
- Rolling historical error: 7-day causal rolling MAE per source (shifted by 1 day)

Evaluated on strict time-based split: Train (Year 1: 2023) -> Test (Year 2: 2024).
Saves trained models to /models/stacking_model.joblib.
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

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING PIPELINE
# -----------------------------------------------------------------------------

def engineer_stacking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs contextual feature matrix for Stage 2 LightGBM model.
    Guarantees strict temporal causality for rolling error features (shift=1).
    """
    df_feat = df.copy()
    df_feat["init_time"] = pd.to_datetime(df_feat["init_time"])
    
    # Sort temporally per spatial grid and forecast stream
    df_feat = df_feat.sort_values(["variable", "lat", "lon", "lead_hours", "init_time"]).reset_index(drop=True)

    # 1. Pairwise differences
    df_feat["diff_1_2"] = df_feat["source_1"] - df_feat["source_2"]
    df_feat["diff_1_3"] = df_feat["source_1"] - df_feat["source_3"]
    df_feat["diff_2_3"] = df_feat["source_2"] - df_feat["source_3"]

    # 2. Ensemble summary statistics
    sources_arr = df_feat[["source_1", "source_2", "source_3"]].values
    df_feat["ens_mean"] = np.mean(sources_arr, axis=1)
    df_feat["ens_spread"] = np.std(sources_arr, axis=1, ddof=1)

    # 3. Seasonal trigonometric encodings
    month = df_feat["init_time"].dt.month
    dayofyear = df_feat["init_time"].dt.dayofyear
    
    df_feat["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df_feat["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    df_feat["sin_doy"] = np.sin(2 * np.pi * dayofyear / 365.25)
    df_feat["cos_doy"] = np.cos(2 * np.pi * dayofyear / 365.25)

    # 4. Weather Regime categorical encoding
    df_feat["weather_regime"] = df_feat["weather_regime"].astype("category")

    # 5. Causal Rolling Historical Error per Source (7-day window shifted by 1)
    grp_cols = ["variable", "lat", "lon", "lead_hours"]
    for src in ["source_1", "source_2", "source_3"]:
        abs_err = (df_feat[src] - df_feat["observation"]).abs()
        df_feat[f"abs_err_{src}"] = abs_err
        
        # Shift by 1 day to ensure zero future data leakage
        df_feat[f"roll_err_{src}"] = df_feat.groupby(grp_cols)[f"abs_err_{src}"].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )
        
        # Fill initial NaNs with baseline group mean
        mean_err = df_feat[f"abs_err_{src}"].mean()
        df_feat[f"roll_err_{src}"] = df_feat[f"roll_err_{src}"].fillna(mean_err)

    return df_feat


# List of feature column names used for training
FEATURE_COLS = [
    "source_1", "source_2", "source_3",
    "diff_1_2", "diff_1_3", "diff_2_3",
    "ens_mean", "ens_spread",
    "lead_hours", "lat", "lon",
    "sin_month", "cos_month", "sin_doy", "cos_doy",
    "weather_regime",
    "roll_err_source_1", "roll_err_source_2", "roll_err_source_3"
]

# -----------------------------------------------------------------------------
# RESIDUAL LIGHTGBM STACKING BLENDER MODEL CLASS
# -----------------------------------------------------------------------------

class LightGBMStackingBlender:
    """
    Stage 2 Context-Aware LightGBM Stacking Blender using Residual Learning.
    Predicts delta = (observation - ensemble_mean), then adds back ensemble_mean.
    """
    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.02, max_depth: int = 5):
        self.name = "5. LightGBM Stacking Blender (Stage 2)"
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.feature_cols = FEATURE_COLS

    def fit(self, df_train: pd.DataFrame):
        for var in ["temperature", "rainfall"]:
            sub = df_train[df_train["variable"] == var].copy()
            X = sub[self.feature_cols]
            # Residual target: delta = observation - ensemble_mean
            y_res = sub["observation"].values - sub["ens_mean"].values

            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=2 ** self.max_depth - 1,
                random_state=42,
                verbosity=-1
            )
            model.fit(X, y_res)
            self.models[var] = model

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df_test))
        
        for var in ["temperature", "rainfall"]:
            mask = df_test["variable"].values == var
            if not np.any(mask):
                continue
            
            sub = df_test[mask]
            X = sub[self.feature_cols]
            ens_mean = sub["ens_mean"].values
            
            model = self.models[var]
            res_preds = model.predict(X)

            # Reconstruct prediction: forecast = ens_mean + residual
            var_preds = ens_mean + res_preds

            if var == "rainfall":
                var_preds = np.maximum(0.0, var_preds) # Non-negative rainfall

            preds[mask] = var_preds
            
        return preds


# -----------------------------------------------------------------------------
# COMPARATIVE EVALUATION ENGINE
# -----------------------------------------------------------------------------

def evaluate_all_stages(
    df_train_feat: pd.DataFrame,
    df_test_feat: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LightGBMStackingBlender]:
    """
    Evaluates Stage 2 LightGBM model against all Stage 1 baselines.
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

    # Train Stage 2 Residual LightGBM Blender
    m_lgbm = LightGBMStackingBlender()
    print("Training Stage 2 Residual LightGBM Regressors (Temperature & Rainfall)...")
    m_lgbm.fit(df_train_feat)

    # Predictions dict
    all_models = {
        "Source 1 (NWP)": df_test_feat["source_1"].values,
        "Source 2 (Ensemble)": df_test_feat["source_2"].values,
        "Source 3 (AI-Model)": df_test_feat["source_3"].values,
        "Persistence": y_pers,
        "1. Equal-Weight Avg": m_eq.predict(df_test_feat),
        "2. Best Single Historical": m_best.predict(df_test_feat),
        "3. Inverse-Error Weighting": m_inv.predict(df_test_feat),
        "4. Bias-Corrected Avg": m_bias.predict(df_test_feat),
        "5. LightGBM Blender (Stage 2)": m_lgbm.predict(df_test_feat)
    }

    # 1. OVERALL RESULTS
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

            overall_rows.append({
                "variable": var,
                "model": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "Skill_Score_RMSE": round(ss_rmse, 4),
                "Skill_Score_MAE": round(ss_mae, 4)
            })

    df_overall = pd.DataFrame(overall_rows)

    # 2. LEAD TIME BREAKDOWN
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

                lead_rows.append({
                    "variable": var,
                    "lead_hours": lead_h,
                    "model": name,
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                    "Skill_Score_RMSE": round(ss_rmse, 4)
                })
    df_lead = pd.DataFrame(lead_rows)

    # 3. SEASON BREAKDOWN
    season_rows = []
    for var in ["temperature", "rainfall"]:
        for season in ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]:
            mask = (df_test_feat["variable"].values == var) & (df_test_feat["season"].values == season)
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

                season_rows.append({
                    "variable": var,
                    "season": season,
                    "model": name,
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                    "Skill_Score_RMSE": round(ss_rmse, 4)
                })
    df_season = pd.DataFrame(season_rows)

    return df_overall, df_lead, df_season, m_lgbm


def main():
    print("Loading synthetic weather dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "weather_blend_bihar_2year.parquet")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")

    df = pd.read_parquet(data_path)

    print("Building Stage 2 features (pairwise differences, ensemble stats, seasonal sin/cos, causal rolling error)...")
    df_feat = engineer_stacking_features(df)

    # STRICT TIME-BASED TRAIN/TEST SPLIT
    # Train: Year 1 (2023) | Test: Year 2 (2024)
    train_mask = df_feat["init_time"] < "2024-01-01"
    df_train = df_feat[train_mask].copy()
    df_test = df_feat[~train_mask].copy()

    print(f"\nTime-Based Train/Test Split:")
    print(f"  - Train Set (2023): {len(df_train):,} rows")
    print(f"  - Test Set  (2024): {len(df_test):,} rows")

    # Train and evaluate Stage 2 vs Stage 1
    df_overall, df_lead, df_season, lgbm_blender = evaluate_all_stages(df_train, df_test)

    # Save trained LightGBM model artifact
    model_save_path = os.path.join(models_dir, "stacking_model.joblib")
    print(f"\nSaving trained Stage 2 LightGBM model to: {model_save_path}")
    joblib.dump(lgbm_blender, model_save_path)

    # Save CSV evaluation metrics
    output_overall_csv = os.path.join(PROJECT_ROOT, "data", "stage2_overall_comparison.csv")
    output_lead_csv = os.path.join(PROJECT_ROOT, "data", "stage2_lead_time_comparison.csv")
    output_season_csv = os.path.join(PROJECT_ROOT, "data", "stage2_season_comparison.csv")

    df_overall.to_csv(output_overall_csv, index=False)
    df_lead.to_csv(output_lead_csv, index=False)
    df_season.to_csv(output_season_csv, index=False)

    # PRINT COMPARISON TABLES
    print("\n" + "=" * 85)
    print("STAGE 2 RESIDUAL STACKING LIGHTGBM vs STAGE 1 BASELINES (TEST SET: YEAR 2 / 2024)")
    print("=" * 85)

    for var in ["temperature", "rainfall"]:
        print(f"\n>>> OVERALL PERFORMANCE - VARIABLE: {var.upper()}")
        print("-" * 85)
        sub = df_overall[df_overall["variable"] == var][["model", "MAE", "RMSE", "Skill_Score_RMSE", "Skill_Score_MAE"]]
        print(sub.to_string(index=False))

    print("\n" + "=" * 85)
    print("MAE BY LEAD TIME (TEMPERATURE °C)")
    print("=" * 85)
    pivot_temp_lead = df_lead[df_lead["variable"] == "temperature"].pivot(index="model", columns="lead_hours", values="MAE")
    print(pivot_temp_lead.round(3).to_string())

    print("\n" + "=" * 85)
    print("MAE BY SEASON (RAINFALL mm)")
    print("=" * 85)
    pivot_rain_season = df_season[df_season["variable"] == "rainfall"].pivot(index="model", columns="season", values="MAE")
    print(pivot_rain_season.round(3).to_string())

    print("\n" + "=" * 85)
    print("LIGHTGBM FEATURE IMPORTANCE TOP 8 (TEMPERATURE & RAINFALL)")
    print("=" * 85)
    for var in ["temperature", "rainfall"]:
        booster = lgbm_blender.models[var].booster_
        fi_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": booster.feature_importance(importance_type="gain")
        }).sort_values("importance", ascending=False)
        print(f"\nTop Features [{var.upper()}]:")
        print(fi_df.head(8).to_string(index=False))

    print(f"\nStage 2 execution complete. Artifacts saved:")
    print(f"  - Model: {model_save_path}")
    print(f"  - Overall Comparison: {output_overall_csv}")
    print(f"  - Lead Time Comparison: {output_lead_csv}")
    print(f"  - Season Comparison: {output_season_csv}")

if __name__ == "__main__":
    main()

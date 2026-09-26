"""
Stage 1 Baseline Models & Evaluation System for Weather Blend MVP (SIH26081)

Implements 4 core baseline blending strategies:
1. Equal-weight ensemble average
2. Best single historical model (lowest training MAE per region/lead-time/season)
3. Inverse-error skill weighting: w_m = (E_m + eps)^-p / sum(w_j)
4. Bias-corrected model average

Evaluated on a strict TIME-BASED train/test split:
- Train: Year 1 (2023)
- Test: Year 2 (2024)
Metrics: MAE, RMSE, and Skill Score vs. Persistence.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# METRICS FUNCTIONS
# -----------------------------------------------------------------------------

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def calculate_skill_score(rmse_model: float, rmse_ref: float) -> float:
    """
    Forecast Skill Score vs. Reference (Persistence).
    SS = 1 - (RMSE_model / RMSE_ref)
    Positive SS implies improvement over reference. 1.0 represents perfect forecast.
    """
    if rmse_ref < 1e-6:
        return 0.0
    return float(1.0 - (rmse_model / rmse_ref))

# -----------------------------------------------------------------------------
# PERSISTENCE LOOKUP HELPER
# -----------------------------------------------------------------------------

def add_persistence_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends 'persistence' observation column to dataframe.
    Persistence prediction for valid_time (t + lead_h) is defined as the observation
    at initialization time t for the given grid point and variable.
    """
    df_out = df.copy()
    
    # Observation lookup by valid_time (which corresponds to observation at that date)
    obs_lookup = df_out[["valid_time", "lat", "lon", "variable", "observation"]].drop_duplicates(
        subset=["valid_time", "lat", "lon", "variable"]
    ).copy()
    obs_lookup = obs_lookup.rename(columns={"valid_time": "init_time", "observation": "persistence"})

    df_out = df_out.merge(obs_lookup, on=["init_time", "lat", "lon", "variable"], how="left")
    
    # Forward fill or fallback for the very first date where init_time persistence is unobserved
    df_out["persistence"] = df_out["persistence"].fillna(df_out["observation"])
    return df_out

# -----------------------------------------------------------------------------
# BASELINE BLENDING MODELS
# -----------------------------------------------------------------------------

class EqualWeightModel:
    """Baseline 1: Simple Equal-Weight Ensemble Average of all 3 sources."""
    def __init__(self):
        self.name = "Equal-Weight Average"

    def fit(self, df_train: pd.DataFrame):
        pass # Parameter-free

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (df["source_1"].values + df["source_2"].values + df["source_3"].values) / 3.0


class BestSingleHistoricalModel:
    """
    Baseline 2: Best Single Historical Model.
    Selects the forecast source with lowest training MAE per context group
    (variable, lead_hours, season).
    """
    def __init__(self, group_cols: List[str] = ["variable", "lead_hours", "season"]):
        self.name = "Best Single Historical"
        self.group_cols = group_cols
        self.best_source_map: Dict[Tuple, str] = {}
        self.overall_best_source: str = "source_1"

    def fit(self, df_train: pd.DataFrame):
        # Overall best fallback
        overall_maes = {
            src: calculate_mae(df_train["observation"].values, df_train[src].values)
            for src in ["source_1", "source_2", "source_3"]
        }
        self.overall_best_source = min(overall_maes, key=overall_maes.get)

        # Per-group best source calculation
        grouped = df_train.groupby(self.group_cols)
        for key, group in grouped:
            maes = {
                src: calculate_mae(group["observation"].values, group[src].values)
                for src in ["source_1", "source_2", "source_3"]
            }
            self.best_source_map[key] = min(maes, key=maes.get)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        # Build key column tuples
        df_keys = list(zip(*[df[col] for col in self.group_cols]))
        
        s1 = df["source_1"].values
        s2 = df["source_2"].values
        s3 = df["source_3"].values

        for idx, key in enumerate(df_keys):
            best_src = self.best_source_map.get(key, self.overall_best_source)
            if best_src == "source_1":
                preds[idx] = s1[idx]
            elif best_src == "source_2":
                preds[idx] = s2[idx]
            else:
                preds[idx] = s3[idx]
        return preds


class InverseErrorWeightingModel:
    """
    Baseline 3: Inverse-Error Skill Weighting.
    Weights w_m = (E_m + eps)^-p / sum((E_j + eps)^-p) computed per context group
    on training set, where E_m is training MAE.
    """
    def __init__(self, group_cols: List[str] = ["variable", "lead_hours", "season"], p: float = 2.0, eps: float = 1e-4):
        self.name = "Inverse-Error Weighting"
        self.group_cols = group_cols
        self.p = p
        self.eps = eps
        self.weights_map: Dict[Tuple, Tuple[float, float, float]] = {}
        self.default_weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)

    def fit(self, df_train: pd.DataFrame):
        grouped = df_train.groupby(self.group_cols)
        for key, group in grouped:
            maes = np.array([
                calculate_mae(group["observation"].values, group[src].values)
                for src in ["source_1", "source_2", "source_3"]
            ])
            raw_weights = (maes + self.eps) ** (-self.p)
            norm_weights = raw_weights / np.sum(raw_weights)
            self.weights_map[key] = tuple(norm_weights)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        s1 = df["source_1"].values
        s2 = df["source_2"].values
        s3 = df["source_3"].values
        
        df_keys = list(zip(*[df[col] for col in self.group_cols]))
        preds = np.zeros(len(df))

        for idx, key in enumerate(df_keys):
            w1, w2, w3 = self.weights_map.get(key, self.default_weights)
            preds[idx] = w1 * s1[idx] + w2 * s2[idx] + w3 * s3[idx]
        return preds


class BiasCorrectedAverageModel:
    """
    Baseline 4: Bias-Corrected Model Average.
    Computes source additive bias b_m = mean(source_m - observation) per group on train set,
    corrects each forecast source, then computes ensemble average.
    """
    def __init__(self, group_cols: List[str] = ["variable", "lead_hours", "season"]):
        self.name = "Bias-Corrected Average"
        self.group_cols = group_cols
        self.bias_map: Dict[Tuple, Tuple[float, float, float]] = {}
        self.default_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def fit(self, df_train: pd.DataFrame):
        grouped = df_train.groupby(self.group_cols)
        for key, group in grouped:
            obs = group["observation"].values
            b1 = float(np.mean(group["source_1"].values - obs))
            b2 = float(np.mean(group["source_2"].values - obs))
            b3 = float(np.mean(group["source_3"].values - obs))
            self.bias_map[key] = (b1, b2, b3)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        s1 = df["source_1"].values
        s2 = df["source_2"].values
        s3 = df["source_3"].values
        vars_arr = df["variable"].values
        
        df_keys = list(zip(*[df[col] for col in self.group_cols]))
        preds = np.zeros(len(df))

        for idx, key in enumerate(df_keys):
            b1, b2, b3 = self.bias_map.get(key, self.default_bias)
            c1 = s1[idx] - b1
            c2 = s2[idx] - b2
            c3 = s3[idx] - b3
            
            # Non-negativity constraint for rainfall
            if vars_arr[idx] == "rainfall":
                c1 = max(0.0, c1)
                c2 = max(0.0, c2)
                c3 = max(0.0, c3)
                
            preds[idx] = (c1 + c2 + c3) / 3.0
        return preds

# -----------------------------------------------------------------------------
# EVALUATION SUITE
# -----------------------------------------------------------------------------

def evaluate_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fits and evaluates all 4 baselines along with individual sources & persistence
    on the test set. Returns overall, lead-time split, and season split evaluation tables.
    """
    # Ensure persistence column exists
    df_test = add_persistence_column(df_test)
    y_true = df_test["observation"].values
    y_pers = df_test["persistence"].values
    rmse_pers_global = calculate_rmse(y_true, y_pers)

    # Initialize baseline models
    models = {
        "Source 1 (NWP)": None,
        "Source 2 (Ensemble)": None,
        "Source 3 (AI-Model)": None,
        "Persistence": None,
        "1. Equal-Weight Avg": EqualWeightModel(),
        "2. Best Single Historical": BestSingleHistoricalModel(),
        "3. Inverse-Error Weighting": InverseErrorWeightingModel(),
        "4. Bias-Corrected Avg": BiasCorrectedAverageModel()
    }

    # Generate predictions dictionary
    predictions = {}
    for name, model in models.items():
        if model is None:
            if name == "Source 1 (NWP)":
                predictions[name] = df_test["source_1"].values
            elif name == "Source 2 (Ensemble)":
                predictions[name] = df_test["source_2"].values
            elif name == "Source 3 (AI-Model)":
                predictions[name] = df_test["source_3"].values
            elif name == "Persistence":
                predictions[name] = y_pers
        else:
            model.fit(df_train)
            predictions[name] = model.predict(df_test)

    # 1. OVERALL EVALUATION TABLE
    overall_rows = []
    for var in ["all", "temperature", "rainfall"]:
        sub_mask = np.ones(len(df_test), dtype=bool) if var == "all" else (df_test["variable"].values == var)
        sub_true = y_true[sub_mask]
        sub_pers = y_pers[sub_mask]
        ref_rmse = calculate_rmse(sub_true, sub_pers)
        ref_mae = calculate_mae(sub_true, sub_pers)

        for name, preds in predictions.items():
            sub_pred = preds[sub_mask]
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

    # 2. LEAD-TIME BREAKDOWN TABLE
    lead_rows = []
    for var in ["temperature", "rainfall"]:
        for lead_h in sorted(df_test["lead_hours"].unique()):
            sub_mask = (df_test["variable"].values == var) & (df_test["lead_hours"].values == lead_h)
            sub_true = y_true[sub_mask]
            sub_pers = y_pers[sub_mask]
            ref_rmse = calculate_rmse(sub_true, sub_pers)

            for name, preds in predictions.items():
                sub_pred = preds[sub_mask]
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

    # 3. SEASON BREAKDOWN TABLE
    season_rows = []
    for var in ["temperature", "rainfall"]:
        for season in ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]:
            sub_mask = (df_test["variable"].values == var) & (df_test["season"].values == season)
            if not np.any(sub_mask):
                continue
            sub_true = y_true[sub_mask]
            sub_pers = y_pers[sub_mask]
            ref_rmse = calculate_rmse(sub_true, sub_pers)

            for name, preds in predictions.items():
                sub_pred = preds[sub_mask]
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

    return df_overall, df_lead, df_season


def main():
    print("Loading weather dataset...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "weather_blend_bihar_2year.parquet")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")

    df = pd.read_parquet(data_path)
    df["init_time"] = pd.to_datetime(df["init_time"])

    # TIME-BASED TRAIN/TEST SPLIT
    # Train: Year 1 (2023) | Test: Year 2 (2024)
    train_mask = df["init_time"] < "2024-01-01"
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()

    print(f"\nStrict Time-Based Split Executed:")
    print(f"  - Train Set (Year 1: 2023): {len(df_train):,} rows ({df_train['init_time'].min().strftime('%Y-%m-%d')} to {df_train['init_time'].max().strftime('%Y-%m-%d')})")
    print(f"  - Test Set  (Year 2: 2024): {len(df_test):,} rows ({df_test['init_time'].min().strftime('%Y-%m-%d')} to {df_test['init_time'].max().strftime('%Y-%m-%d')})")

    # Evaluate models
    df_overall, df_lead, df_season = evaluate_models(df_train, df_test)

    # Save output tables
    output_overall_csv = os.path.join(project_root, "data", "baseline_overall_results.csv")
    output_lead_csv = os.path.join(project_root, "data", "baseline_lead_time_results.csv")
    output_season_csv = os.path.join(project_root, "data", "baseline_season_results.csv")

    df_overall.to_csv(output_overall_csv, index=False)
    df_lead.to_csv(output_lead_csv, index=False)
    df_season.to_csv(output_season_csv, index=False)

    # PRINT SUMMARY COMPARISON TABLES
    print("\n" + "=" * 80)
    print("STAGE 1 BASELINE EVALUATION SUMMARY (TEST SET: YEAR 2 / 2024)")
    print("=" * 80)

    for var in ["temperature", "rainfall"]:
        print(f"\n>>> OVERALL PERFORMANCE - VARIABLE: {var.upper()}")
        print("-" * 80)
        sub = df_overall[df_overall["variable"] == var][["model", "MAE", "RMSE", "Skill_Score_RMSE", "Skill_Score_MAE"]]
        print(sub.to_string(index=False))

    print("\n" + "=" * 80)
    print("MAE BY LEAD TIME (TEMPERATURE °C)")
    print("=" * 80)
    pivot_temp_lead = df_lead[df_lead["variable"] == "temperature"].pivot(index="model", columns="lead_hours", values="MAE")
    print(pivot_temp_lead.round(3).to_string())

    print("\n" + "=" * 80)
    print("MAE BY SEASON (RAINFALL mm)")
    print("=" * 80)
    pivot_rain_season = df_season[df_season["variable"] == "rainfall"].pivot(index="model", columns="season", values="MAE")
    print(pivot_rain_season.round(3).to_string())

    print("\n" + "=" * 80)
    print("SKILL SCORE (vs. Persistence) BY LEAD TIME (TEMPERATURE)")
    print("=" * 80)
    pivot_temp_ss = df_lead[df_lead["variable"] == "temperature"].pivot(index="model", columns="lead_hours", values="Skill_Score_RMSE")
    print(pivot_temp_ss.round(3).to_string())

    print(f"\nEvaluation complete. Results saved to:\n  - {output_overall_csv}\n  - {output_lead_csv}\n  - {output_season_csv}")

if __name__ == "__main__":
    main()

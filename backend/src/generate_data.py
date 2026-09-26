"""
Synthetic Data Generator for Weather Blend MVP (SIH26081)
Simulates 3 forecast sources (NWP, Ensemble, AI-Model) for a 5x5 grid over Bihar, India.
Includes realistic ground-truth observations, error profiles per source/season/lead-time,
and weather regime clustering.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Set seed for reproducibility
np.random.seed(42)

def get_season(month: int) -> str:
    """Return season name based on month in India/Bihar climate context."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else: # 10, 11
        return "Post-Monsoon"

def generate_synthetic_weather_dataset(
    start_date: str = "2023-01-01",
    days: int = 730,
    lats: list = [24.5, 25.2, 25.9, 26.6, 27.3],
    lons: list = [83.5, 84.5, 85.5, 86.5, 87.5],
    lead_hours_list: list = [24, 48, 72, 120]
) -> pd.DataFrame:
    """
    Generates a 2-year daily forecast dataset over a 5x5 grid in Bihar.
    
    Error Profiles:
    - NWP (source_1): Physics model, highly accurate at 24h, rapidly degrades at 120h.
    - Ensemble (source_2): Multi-model ensemble, performs best in Monsoon & volatile regimes.
    - AI-Model (source_3): Deep learning forecast, performs best at long lead times (72h, 120h).
    """
    init_dates = pd.date_range(start=start_date, periods=days, freq="D")
    records = []

    # Pre-generate daily regional weather driver (temporal AR(1) state)
    temp_synoptic = np.zeros(days)
    t_curr = 0.0
    for i in range(days):
        t_curr = 0.85 * t_curr + np.random.normal(0, 1.2)
        temp_synoptic[i] = t_curr

    for day_idx, init_dt in enumerate(init_dates):
        day_of_year = init_dt.dayofyear
        month = init_dt.month
        season = get_season(month)

        # Baseline ground truth temperature (annual sine wave)
        base_temp_day = 24.5 - 11.0 * np.cos(2 * np.pi * (day_of_year + 15) / 365.0) + temp_synoptic[day_idx]

        # Baseline ground truth rainfall probability & intensity
        if season == "Monsoon":
            p_rain = 0.70
            rain_scale = 22.0
        elif season == "Pre-Monsoon":
            p_rain = 0.25
            rain_scale = 6.0
        elif season == "Post-Monsoon":
            p_rain = 0.15
            rain_scale = 4.0
        else: # Winter
            p_rain = 0.05
            rain_scale = 2.0

        for lat in lats:
            for lon in lons:
                # Spatial gradient effects across Bihar grid
                spatial_temp_offset = -0.4 * (lat - 24.5) + 0.2 * (lon - 83.5)
                grid_temp_obs = base_temp_day + spatial_temp_offset + np.random.normal(0, 0.4)
                grid_temp_obs = float(np.clip(grid_temp_obs, 4.0, 46.0))

                # Grid ground truth rain
                is_raining = np.random.rand() < p_rain
                if is_raining:
                    grid_rain_obs = float(np.random.gamma(shape=1.4, scale=rain_scale) + np.random.uniform(0, 3.0))
                else:
                    grid_rain_obs = 0.0

                for lead_h in lead_hours_list:
                    valid_dt = init_dt + pd.Timedelta(hours=lead_h)
                    lead_factor = lead_h / 24.0 # 1.0, 2.0, 3.0, 5.0

                    # -------------------------------------------------------------
                    # VARIABLE 1: TEMPERATURE (°C)
                    # -------------------------------------------------------------
                    # NWP (source_1): low error at 24h (~0.5C), high degradation at 120h (~4.5C)
                    nwp_t_std = 0.6 * (lead_factor ** 1.35)
                    nwp_t_bias = 0.2 * (lead_factor - 1)
                    s1_temp = grid_temp_obs + nwp_t_bias + np.random.normal(0, nwp_t_std)

                    # Ensemble (source_2): moderate flat error (1.1 + 0.3 * lead), best in Monsoon volatility
                    ens_monsoon_discount = 0.7 if season == "Monsoon" else 1.0
                    ens_t_std = (1.1 + 0.30 * lead_factor) * ens_monsoon_discount
                    ens_t_bias = -0.1
                    s2_temp = grid_temp_obs + ens_t_bias + np.random.normal(0, ens_t_std)

                    # AI-Model (source_3): baseline 1.15C, very slow growth (1.15 + 0.12 * lead)
                    ai_t_std = 1.15 + 0.12 * lead_factor
                    ai_t_bias = 0.05
                    s3_temp = grid_temp_obs + ai_t_bias + np.random.normal(0, ai_t_std)

                    records.append({
                        "init_time": init_dt,
                        "valid_time": valid_dt,
                        "lead_hours": lead_h,
                        "lat": lat,
                        "lon": lon,
                        "variable": "temperature",
                        "source_1": round(float(s1_temp), 2),
                        "source_2": round(float(s2_temp), 2),
                        "source_3": round(float(s3_temp), 2),
                        "observation": round(float(grid_temp_obs), 2),
                        "month": month,
                        "season": season
                    })

                    # -------------------------------------------------------------
                    # VARIABLE 2: RAINFALL (mm)
                    # -------------------------------------------------------------
                    # NWP (source_1): struggles at long lead times and monsoon convection
                    nwp_r_std = (1.5 + 2.5 * (lead_factor ** 1.2)) * (1.4 if season == "Monsoon" else 1.0)
                    s1_rain = max(0.0, grid_rain_obs + np.random.normal(0, nwp_r_std))

                    # Ensemble (source_2): excels during Monsoon convective extremes
                    ens_r_mult = 0.65 if season == "Monsoon" else 1.0
                    ens_r_std = (2.2 + 1.1 * lead_factor) * ens_r_mult
                    s2_rain = max(0.0, grid_rain_obs + np.random.normal(0, ens_r_std))

                    # AI-Model (source_3): excels at 72h and 120h lead times for spatial precipitation
                    ai_r_std = 2.0 + 0.7 * lead_factor
                    s3_rain = max(0.0, grid_rain_obs + np.random.normal(0, ai_r_std))

                    records.append({
                        "init_time": init_dt,
                        "valid_time": valid_dt,
                        "lead_hours": lead_h,
                        "lat": lat,
                        "lon": lon,
                        "variable": "rainfall",
                        "source_1": round(float(s1_rain), 2),
                        "source_2": round(float(s2_rain), 2),
                        "source_3": round(float(s3_rain), 2),
                        "observation": round(float(grid_rain_obs), 2),
                        "month": month,
                        "season": season
                    })

    df = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # WEATHER REGIME CLUSTERING (Pivoted by Location & Init Date)
    # -------------------------------------------------------------------------
    pivoted = df.pivot_table(
        index=["init_time", "lat", "lon", "month", "season"],
        columns="variable",
        values="observation",
        aggfunc="first"
    ).reset_index()

    features = pivoted[["temperature", "rainfall", "month"]].copy()
    feat_norm = (features - features.mean()) / (features.std() + 1e-6)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    pivoted["cluster_id"] = kmeans.fit_predict(feat_norm)

    centroids = pivoted.groupby("cluster_id")[["temperature", "rainfall"]].mean()
    
    regime_names = {}
    for cid in range(4):
        t_val = centroids.loc[cid, "temperature"]
        r_val = centroids.loc[cid, "rainfall"]
        
        if r_val > 10.0:
            regime_names[cid] = "Monsoon_Convective"
        elif t_val > 30.0:
            regime_names[cid] = "Hot_Dry_PreMonsoon"
        elif t_val < 18.0:
            regime_names[cid] = "Cool_Dry_Winter"
        else:
            regime_names[cid] = "Moderate_Transitional"

    pivoted["weather_regime"] = pivoted["cluster_id"].map(regime_names)

    df = df.merge(
        pivoted[["init_time", "lat", "lon", "weather_regime"]],
        on=["init_time", "lat", "lon"],
        how="left"
    )

    cols_order = [
        "init_time", "valid_time", "lead_hours", "lat", "lon",
        "variable", "source_1", "source_2", "source_3",
        "observation", "season", "weather_regime"
    ]
    df = df[cols_order]
    return df

def main():
    print("Generating synthetic 2-year weather dataset for Bihar region...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = generate_synthetic_weather_dataset()

    parquet_path = os.path.join(data_dir, "weather_blend_bihar_2year.parquet")
    csv_path = os.path.join(data_dir, "weather_blend_bihar_2year.csv")

    print(f"Saving parquet to: {parquet_path}")
    df.to_parquet(parquet_path, index=False)

    print(f"Saving CSV to: {csv_path}")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("DATASET GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total Rows Generated: {len(df):,}")
    print(f"Schema Columns: {list(df.columns)}")
    print(f"Unique Init Dates: {df['init_time'].nunique()} ({df['init_time'].min().strftime('%Y-%m-%d')} to {df['init_time'].max().strftime('%Y-%m-%d')})")
    print(f"Grid Points: {df[['lat', 'lon']].drop_duplicates().shape[0]} (5x5 grid over Bihar)")
    print(f"Lead Times: {sorted(df['lead_hours'].unique())} hours")
    print(f"Variables: {list(df['variable'].unique())}")
    print(f"Seasons: {dict(df['season'].value_counts())}")
    print(f"Weather Regimes: {dict(df['weather_regime'].value_counts())}")

if __name__ == "__main__":
    main()

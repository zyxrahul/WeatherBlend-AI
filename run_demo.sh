#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "          WEATHER BLEND MVP (SIH26081) - ONE-COMMAND DEMO PIPELINE"
echo "======================================================================"
echo ""

echo "Step 1/5: Installing / Verifying Dependencies..."
pip install -r requirements.txt
echo ""

echo "Step 2/5: Generating Synthetic Dataset (Backend Engine - Bihar 5x5 Grid)..."
python backend/src/generate_data.py
echo ""

echo "Step 3/5: Evaluating Stage 1 Baselines..."
python backend/src/baselines.py
echo ""

echo "Step 4/5: Training Stage 2 LightGBM Stacking Model..."
python backend/src/stacking_model.py
echo ""

echo "Step 5/5: Training Stage 3 Adaptive Gating & Two-Stage Model..."
python backend/src/gating_model.py
echo ""

echo "======================================================================"
echo "PIPELINE COMPLETE! LAUNCHING FRONTEND STREAMLIT DASHBOARD..."
echo "======================================================================"
streamlit run frontend/app.py

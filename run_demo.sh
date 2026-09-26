#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "          WEATHER BLEND MVP (SIH26081) - ONE-COMMAND DEMO PIPELINE"
echo "======================================================================"
echo ""

echo "Step 1/5: Installing / Verifying Dependencies..."
pip install -r requirements.txt
echo ""

echo "Step 2/5: Generating Synthetic Dataset (Bihar 5x5 Grid, 2 Years)..."
python src/generate_data.py
echo ""

echo "Step 3/5: Evaluating Stage 1 Baselines..."
python src/baselines.py
echo ""

echo "Step 4/5: Training Stage 2 LightGBM Stacking Model..."
python src/stacking_model.py
echo ""

echo "Step 5/5: Training Stage 3 Adaptive Gating & Two-Stage Model..."
python src/gating_model.py
echo ""

echo "======================================================================"
echo "PIPELINE COMPLETE! LAUNCHING INTERACTIVE STREAMLIT DASHBOARD..."
echo "======================================================================"
streamlit run dashboard/app.py

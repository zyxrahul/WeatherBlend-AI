@echo off
echo ======================================================================
echo           WEATHER BLEND MVP (SIH26081) - ONE-COMMAND DEMO PIPELINE
echo ======================================================================
echo.

echo Step 1/5: Installing / Verifying Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies! Exiting.
    exit /b %errorlevel%
)
echo.

echo Step 2/5: Generating Synthetic Dataset (Bihar 5x5 Grid, 2 Years)...
python src\generate_data.py
if %errorlevel% neq 0 (
    echo Error generating data! Exiting.
    exit /b %errorlevel%
)
echo.

echo Step 3/5: Evaluating Stage 1 Baselines...
python src\baselines.py
if %errorlevel% neq 0 (
    echo Error evaluating baselines! Exiting.
    exit /b %errorlevel%
)
echo.

echo Step 4/5: Training Stage 2 LightGBM Stacking Model...
python src\stacking_model.py
if %errorlevel% neq 0 (
    echo Error training Stage 2 model! Exiting.
    exit /b %errorlevel%
)
echo.

echo Step 5/5: Training Stage 3 Adaptive Gating & Two-Stage Model...
python src\gating_model.py
if %errorlevel% neq 0 (
    echo Error training Stage 3 model! Exiting.
    exit /b %errorlevel%
)
echo.

echo ======================================================================
echo PIPELINE COMPLETE! LAUNCHING INTERACTIVE STREAMLIT DASHBOARD...
echo ======================================================================
streamlit run dashboard\app.py

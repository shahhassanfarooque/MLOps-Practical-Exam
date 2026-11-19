@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo           Part 5: Unified End-to-End MLOps Workflow Runner
echo ======================================================================
echo.

:: 1. Check/Set MLflow Tracking URI
if "%MLFLOW_TRACKING_URI%"=="" (
    set "MLFLOW_TRACKING_URI=http://127.0.0.1:5000"
)
echo [*] MLflow Tracking Server URI: %MLFLOW_TRACKING_URI%

:: 2. Activate Virtual Environment
if exist "..\envs\.venv-e2e\Scripts\activate.bat" (
    echo [*] Activating unified environment: envs\.venv-e2e...
    call ..\envs\.venv-e2e\Scripts\activate.bat
) else if exist "..\envs\.venv-mlflow\Scripts\activate.bat" (
    echo [*] Activating fallback environment: envs\.venv-mlflow...
    call ..\envs\.venv-mlflow\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environments not found! Please run setup_envs.bat in the workspace root first.
    exit /b 1
)

echo.
echo ======================================================================
echo [Stage 1/3] Data Ingestion, Imputation, Feature Engineering & Validation
echo ======================================================================
python src\stage1_prepare_data.py --output data\processed\validated_features.parquet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Stage 1 failed! Halting pipeline execution.
    exit /b 1
)

echo.
echo ======================================================================
echo [Stage 2/3] Feast Feature Store Ingestion, Materialization & Point-in-Time Join
echo ======================================================================
python feature_repo\stage2_feast_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Stage 2 failed! Halting pipeline execution.
    exit /b 1
)

echo.
echo ======================================================================
echo [Stage 3/3] MLflow Experiment Tracking, Model Selection, Registry & Live Serving
echo ======================================================================
python src\stage3_train_register_infer.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Stage 3 failed! Halting pipeline execution.
    exit /b 1
)

echo.
echo ======================================================================
echo [SUCCESS] End-to-End MLOps Pipeline Completed Successfully!
echo ======================================================================

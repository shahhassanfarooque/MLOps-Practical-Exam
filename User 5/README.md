# Part 5: Unified End-to-End MLOps Workflow (`part-5-e2e-integration`)

## 1. Exam Context & Group Collaboration Workflow
- **Exam Repository:** `https://github.com/<owner>/MLOps-Practical-Exam.git`
- **Assigned Branch:** `part-5-e2e-integration`
- **Target Runtime Dataset:** Selected by the examiner (`"wine"`, `"breast_cancer"`, or `"digits"`).

### Exam Submission Protocol (Follow in `cmd.exe`):
1. **Clone the shared repository:**
   ```cmd
   git clone https://github.com/<owner>/MLOps-Practical-Exam.git
   cd MLOps-Practical-Exam
   ```
2. **Sync `main` before branching:**
   ```cmd
   git switch main
   git pull origin main
   ```
3. **Create dedicated branch:**
   ```cmd
   git switch -c part-5-e2e-integration
   ```
4. **Copy workspace folder:** Copy the completed `part-5-e2e-integration\` directory contents into your cloned repository branch.
5. **Verify runtime dataset:** Check `dataset_loader.py` matches the assigned dataset name.
6. **Commit & Push using Conventional Commits:**
   ```cmd
   git add .
   git commit -m "feat(e2e): implement unified dvc feast mlflow closed loop mlops workflow"
   git push -u origin part-5-e2e-integration
   ```
7. **Submit via Pull Request:** Open PR from `compare: part-5-e2e-integration` to `base: main` on GitHub.

---

## 2. Step-by-Step Execution Guide (Windows `cmd.exe`)

The end-to-end integration unites DVC, Feast, and MLflow into a single automated pipeline.

### Terminal 1: Launch MLflow Tracking Server
1. Open Command Prompt and navigate into `part-5-e2e-integration`:
   ```cmd
   cd "part-5-e2e-integration"
   ```
2. Activate the MLflow virtual environment:
   ```cmd
   call ..\envs\.venv-mlflow\Scripts\activate.bat
   ```
3. Start the server:
   ```cmd
   mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
   ```
   *(Keep this terminal running. Access UI at `http://127.0.0.1:5000`)*

---

### Terminal 2: Execute Unified Pipeline
1. Open a second Command Prompt and navigate into `part-5-e2e-integration`:
   ```cmd
   cd "part-5-e2e-integration"
   ```
2. Select runtime dataset in `dataset_loader.py`:
   ```python
   # Options: "wine", "breast_cancer", "digits"
   DATASET_NAME = "wine"
   ```
3. **Execute One-Click E2E Runner:**
   ```cmd
   run_e2e.bat
   ```
   *Alternatively, if using DVC directly:*
   ```cmd
   call ..\envs\.venv-e2e\Scripts\activate.bat
   dvc init --no-scm
   dvc repro
   ```

### Execution Flow & Verification Checklist:
- **Stage 1 (Data Prep):** Validates raw data, creates `data\processed\validated_features.parquet`, enforces zero-null and non-inf assertions.
- **Stage 2 (Feast Store):** Generates `features.py`, runs `feast apply`, materializes SQLite online store (`feature_repo\data\online_store.db`), and builds point-in-time training set `data\processed\feast_training_set.parquet`.
- **Stage 3 (MLflow & Serving):** Trains 3 models, logs parameters/metrics/plots to MLflow, registers top model as `"e2e-classifier-prod"` in `Staging`, queries live online features from SQLite, and generates real-time predictions.

---

## 3. Reset to Initial Clean State

> [!WARNING]
> Before running cleanup, switch to **Terminal 1** and press `Ctrl+C` to terminate the running MLflow server, unlocking `mlflow.db`.

```cmd
del /q "data\processed\*.parquet" 2>nul
del /q "data\processed\*.csv" 2>nul
del /q "feature_repo\data\*.parquet" 2>nul
del /q "feature_repo\data\*.db" 2>nul
del /q "feature_repo\data\*.db-journal" 2>nul
del /q "feature_repo\features.py" 2>nul
del /q "mlflow.db" 2>nul
del /q "confusion_matrix_*.png" 2>nul
del /q "dvc.lock" 2>nul
del /q ".dvcignore" 2>nul
if exist ".dvc" rmdir /s /q ".dvc"
if exist "mlruns" rmdir /s /q "mlruns"
if exist "mlartifacts" rmdir /s /q "mlartifacts"
for /d /r . %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
```

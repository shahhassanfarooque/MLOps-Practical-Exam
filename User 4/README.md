# Part 4: MLflow Experiment Tracking & Model Registry (`part-4-mlflow`)

## 1. Exam Context & Group Collaboration Workflow
- **Exam Repository:** `https://github.com/<owner>/MLOps-Practical-Exam.git`
- **Assigned Branch:** `part-4-mlflow`
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
   git switch -c part-4-mlflow
   ```
4. **Copy workspace folder:** Copy the completed `part-4-mlflow\` directory contents into your cloned repository branch.
5. **Verify runtime dataset:** Check `dataset_loader.py` matches the assigned dataset name.
6. **Commit & Push using Conventional Commits:**
   ```cmd
   git add .
   git commit -m "feat(mlflow): implement experiment tracking, model comparison, and registry staging"
   git push -u origin part-4-mlflow
   ```
7. **Submit via Pull Request:** Open PR from `compare: part-4-mlflow` to `base: main` on GitHub.

---

## 2. Step-by-Step Execution Guide (Windows 2-Terminal Workflow)

MLflow operates with a persistent centralized tracking server backed by a local SQLite database (`sqlite:///mlflow.db`). Two Windows Command Prompt (`cmd.exe`) windows are used.

### Terminal 1: Start MLflow Tracking Server
1. Open a new Command Prompt and navigate into `part-4-mlflow`:
   ```cmd
   cd "part-4-mlflow"
   ```
2. Activate the MLflow virtual environment:
   ```cmd
   call ..\envs\.venv-mlflow\Scripts\activate.bat
   ```
3. Launch the MLflow server:
   ```cmd
   mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
   ```
   *(Keep this terminal running. The MLflow Web UI is now accessible at `http://127.0.0.1:5000`)*

---

### Terminal 2: Train, Compare, Register & Infer
1. Open a second Command Prompt and navigate into `part-4-mlflow`:
   ```cmd
   cd "part-4-mlflow"
   ```
2. Activate the MLflow virtual environment:
   ```cmd
   call ..\envs\.venv-mlflow\Scripts\activate.bat
   ```
3. Configure the tracking server URL:
   ```cmd
   set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   ```
4. Select the runtime dataset in `dataset_loader.py`:
   ```python
   # Options: "wine", "breast_cancer", "digits"
   DATASET_NAME = "wine"
   ```
5. **Run Experiment Tracking & Model Training:**
   ```cmd
   python src/train_with_mlflow.py
   ```
   *Trains LogisticRegression (with StandardScaler for convergence safety), Shallow RandomForest, and Deep RandomForest.*
   *Logs parameters, macro metrics, confusion matrix plots, and MLmodel serialization.*
   *Prints side-by-side comparison table in console.*

6. **Register the Best Performing Model:**
   ```cmd
   python src/register_best_model.py
   ```
   *Automatically queries the run with the highest `f1_macro`, registers it as `"exam-classifier-prod"`, and transitions it to `"Staging"`.*

7. **Load Registered Staging Model & Run Inference:**
   ```cmd
   python src/load_registered_model.py
   ```
   *Loads `models:/exam-classifier-prod/Staging` and generates live predictions on test samples.*

---

## 3. Web UI Inspection
Open your browser to `http://127.0.0.1:5000`:
- **Experiments View:** Select `"exam-classification-baseline"` to compare accuracy, precision, recall, and F1 across runs.
- **Artifacts Tab:** Click on each run to view `evaluation_plots/confusion_matrix_<model>.png` and the packaged `MLmodel` schema.
- **Models Tab:** View registered model `"exam-classifier-prod"` and verify Version 1 is marked as **Staging**.

---

## 4. Reset to Initial Clean State

> [!WARNING]
> Before running cleanup, go to **Terminal 1** and press `Ctrl+C` to terminate the running MLflow server, unlocking `mlflow.db`.

```cmd
del /q "mlflow.db" 2>nul
del /q "confusion_matrix_*.png" 2>nul
if exist "mlruns" rmdir /s /q "mlruns"
if exist "mlartifacts" rmdir /s /q "mlartifacts"
for /d /r . %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
```

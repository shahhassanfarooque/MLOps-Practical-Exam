# Part 2: Automated Machine Learning Pipeline (`part-2-pipeline`)

## 1. Exam Context & Group Collaboration Workflow
- **Exam Repository:** `https://github.com/<owner>/MLOps-Practical-Exam.git`
- **Assigned Branch:** `part-2-pipeline`
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
   git switch -c part-2-pipeline
   ```
4. **Copy workspace folder:** Copy the completed `part-2-pipeline\` directory contents into your cloned repository branch.
5. **Verify runtime dataset:** Check `dataset_loader.py` matches the assigned dataset name.
6. **Commit & Push using Conventional Commits:**
   ```cmd
   git add .
   git commit -m "feat(pipeline): implement modular 5-stage dvc reproducible pipeline"
   git push -u origin part-2-pipeline
   ```
7. **Submit via Pull Request:** Open PR from `compare: part-2-pipeline` to `base: main` on GitHub.

---

## 2. Step-by-Step Execution Guide (Windows `cmd.exe`)

Open Command Prompt (`cmd.exe`) and navigate into `part-2-pipeline`:
```cmd
cd "part-2-pipeline"
```

### Step 2.1: Activate Virtual Environment
```cmd
call ..\envs\.venv-base\Scripts\activate.bat
```
*(Verify prompt shows `(.venv-base)`)*

### Step 2.2: Select Runtime Dataset
Open `dataset_loader.py` and set the assigned dataset:
```python
# Options: "wine", "breast_cancer", "digits"
DATASET_NAME = "wine"
```

### Step 2.3: Initialize DVC
- If running inside your cloned Git repository:
  ```cmd
  dvc init
  ```
- If testing locally standalone without Git:
  ```cmd
  dvc init --no-scm
  ```

### Step 2.4: Execute Pipeline (`dvc repro`)
Run all 5 stages of the pipeline:
```cmd
dvc repro
```
*DVC executes: `collect` -> `preprocess` -> `features` -> `validate` -> `train_eval`.*

### Step 2.5: Verify Pipeline Caching
Run `dvc repro` a second time:
```cmd
dvc repro
```
*Observe that DVC reports all stages skipped because inputs and code hashes are unchanged:*
```
Stage 'collect' didn't change, skipping
Stage 'preprocess' didn't change, skipping
Stage 'features' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'train_eval' didn't change, skipping
Data and pipelines are up to date.
```

### Step 2.6: Inspect Pipeline DAG and Metrics
View the ASCII Directed Acyclic Graph:
```cmd
dvc dag
```
Inspect evaluation metrics:
```cmd
dvc metrics show
```

### Step 2.7: Test Validation Hard-Stop (`sys.exit(1)`)
To test that Stage 4 halts execution when invalid data is supplied:
```cmd
python src/pipeline/validate.py --input src/pipeline/validate.py
```
*Observe the hard stop error and exit code:*
```
[VALIDATION FAILURE] DataValidationError: Target column 'target_label' is missing from the dataset!
[HARD STOP] Halting ML pipeline execution due to data quality gate failure.
```
```cmd
echo %ERRORLEVEL%
```
*(Prints `1`)*

---

## 3. Reset to Initial Clean State

To cleanly remove all generated CSVs, models, metrics, and DVC lockfiles before testing another dataset:
```cmd
del /q "data\raw\*.csv" 2>nul
del /q "data\processed\*.csv" 2>nul
del /q "models\*.joblib" 2>nul
del /q "metrics\*.json" 2>nul
del /q "dvc.lock" 2>nul
del /q ".dvcignore" 2>nul
if exist ".dvc" rmdir /s /q ".dvc"
for /d /r . %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
```

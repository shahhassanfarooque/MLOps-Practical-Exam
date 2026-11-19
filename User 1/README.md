# Part 1: Data Versioning Using DVC (`part-1-dvc`)

## 1. Exam Context & Group Collaboration Workflow
- **Exam Repository:** `https://github.com/<owner>/MLOps-Practical-Exam.git`
- **Assigned Branch:** `part-1-dvc`
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
   git switch -c part-1-dvc
   ```
4. **Copy workspace folder:** Copy the completed `part-1-dvc\` directory contents into your cloned repository branch.
5. **Verify runtime dataset:** Check `dataset_loader.py` matches the assigned dataset name.
6. **Commit & Push using Conventional Commits:**
   ```cmd
   git add .
   git commit -m "feat(dvc): implement data versioning and remote tracking"
   git push -u origin part-1-dvc
   ```
7. **Submit via Pull Request:** Open PR from `compare: part-1-dvc` to `base: main` on GitHub.

---

## 2. Step-by-Step Execution Guide (Windows `cmd.exe`)

Open Command Prompt (`cmd.exe`) and navigate into the `part-1-dvc` folder:
```cmd
cd "part-1-dvc"
```

### Step 2.1: Activate Virtual Environment
Activate the base environment configured with DVC:
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

### Step 2.3: Generate Version 1 Data
```cmd
python src/generate_data.py
```
*Expected output: Generates `data\raw\dataset_v1.csv` and prints exact row/feature counts.*

### Step 2.4: Initialize DVC
- If running inside your cloned Git repository:
  ```cmd
  dvc init
  ```
- If testing locally standalone without Git:
  ```cmd
  dvc init --no-scm
  ```

### Step 2.5: Configure Local Remote Storage
Set up local remote storage outside the repo (e.g. in your user profile):
```cmd
dvc remote add -d myremote "%USERPROFILE%\dvc-remote-storage"
```
*(Verify with `dvc remote list`)*

### Step 2.6: Track & Push Version 1 to Remote
```cmd
dvc add data\raw\dataset_v1.csv
dvc push
```
*DVC creates `data\raw\dataset_v1.csv.dvc` containing the MD5 checksum and uploads the cached blob to the remote storage.*

### Step 2.7: Augment Data to Create Version 2
```cmd
python src/augment_data.py
```
*Appends 20 Gaussian-perturbed rows (seed=42) and updates `data\raw\dataset_v1.csv`.*

### Step 2.8: Track Version 2 and Compare Versions
```cmd
dvc add data\raw\dataset_v1.csv
dvc diff
dvc push
```
*`dvc diff` reports the modified file and size difference between commits/versions.*

### Step 2.9: Restore Historical Version 1
To demonstrate dataset restoration:
1. Copy or revert the pointer file `data\raw\dataset_v1.csv.dvc` (or checkout from Git).
2. Run DVC checkout:
   ```cmd
   dvc checkout
   ```
3. Run `python -c "import pandas as pd; print('Current rows:', len(pd.read_csv('data/raw/dataset_v1.csv')))"` to confirm the original row count is restored in milliseconds!

---

## 3. Reset to Initial Clean State

To cleanly remove all generated CSVs, `.dvc` files, and local DVC cache before testing another dataset or submitting:
```cmd
del /q "data\raw\*.csv" 2>nul
del /q "data\raw\*.dvc" 2>nul
del /q "dvc.lock" 2>nul
del /q ".dvcignore" 2>nul
if exist ".dvc" rmdir /s /q ".dvc"
for /d /r . %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
```

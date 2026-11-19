# Part 3: Feature Store Implementation with Feast (`part-3-feast`)

## 1. Exam Context & Group Collaboration Workflow
- **Exam Repository:** `https://github.com/<owner>/MLOps-Practical-Exam.git`
- **Assigned Branch:** `part-3-feast`
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
   git switch -c part-3-feast
   ```
4. **Copy workspace folder:** Copy the completed `part-3-feast\` directory contents into your cloned repository branch.
5. **Verify runtime dataset:** Check `dataset_loader.py` matches the assigned dataset name.
6. **Commit & Push using Conventional Commits:**
   ```cmd
   git add .
   git commit -m "feat(feast): implement feature store with sqlite online and parquet offline stores"
   git push -u origin part-3-feast
   ```
7. **Submit via Pull Request:** Open PR from `compare: part-3-feast` to `base: main` on GitHub.

---

## 2. Step-by-Step Execution Guide (Windows `cmd.exe`)

Open Command Prompt (`cmd.exe`) and navigate into `part-3-feast\feature_repo`:
```cmd
cd "part-3-feast\feature_repo"
```

### Step 2.1: Activate Virtual Environment
```cmd
call ..\..\envs\.venv-feast\Scripts\activate.bat
```
*(Verify prompt shows `(.venv-feast)`)*

### Step 2.2: Select Runtime Dataset
Open `..\dataset_loader.py` and set the assigned dataset:
```python
# Options: "wine", "breast_cancer", "digits"
DATASET_NAME = "wine"
```

### Step 2.3: Ingest Data & Generate Feature Definitions
Run the setup script to extract features, build Parquet offline store, and dynamically generate `features.py`:
```cmd
python setup_feature_store.py
```
*Expected output: Generates `data\features.parquet` and `features.py` containing Entity, Sources, FeatureViews, and FeatureService.*

### Step 2.4: Register Feature Definitions (`feast apply`)
Register entities and feature views into the Feast registry database (`data\registry.db`):
```cmd
feast apply
```

### Step 2.5: Verify Registered Feature Store Schema
Inspect registered Feast objects:
```cmd
feast entities list
feast feature-views list
feast feature-services list
```

### Step 2.6: Materialize Features into SQLite Online Store
Run the cross-platform materialization wrapper:
```cmd
python materialize.py
```
*Feast synchronizes offline features up to the current UTC timestamp into `data\online_store.db`.*

### Step 2.7: Retrieve Low-Latency Online Features
Query the SQLite online store for inference samples:
```cmd
python get_online_features.py
```
*Demonstrates sub-10ms feature retrieval for entity IDs 0, 1, and 2.*

### Step 2.8: Retrieve Point-in-Time Correct Historical Features
Generate training dataset via point-in-time joins:
```cmd
python get_historical_features.py
```
*Demonstrates leakage-free historical feature joins.*

### Step 2.9: Demonstrate Cross-Model Feature Reusability
Train an unsupervised KMeans clustering model using the shared feature service:
```cmd
python reuse_features_clustering.py
```
*Demonstrates that multiple models and teams can consume identical features with zero duplicate feature engineering code.*

---

## 3. Reset to Initial Clean State

To cleanly remove all generated Parquet files, SQLite databases, and dynamic `features.py` before testing another dataset:
```cmd
del /q "data\*.parquet" 2>nul
del /q "data\*.db" 2>nul
del /q "data\*.db-journal" 2>nul
del /q "features.py" 2>nul
for /d /r . %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d" 2>nul
```

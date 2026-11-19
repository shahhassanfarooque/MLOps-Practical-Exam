# DVC Data Versioning Workflow Guide

This document details the architecture, theory, and operational mechanics of **Data Version Control (DVC)** for dataset tracking, storage management, and historical version restoration.

---

## 1. Why Data Version Control (DVC)?

In production machine learning workflows:
- Code is small (KB to MB) and versioned with **Git**.
- Data and models are large (MB to GB/TB) and cannot be tracked directly in Git repositories due to repository bloat, slow cloning times, and file size limits (GitHub 100MB limit).
- **DVC solves this separation of concerns** by acting as a bridge:
  - DVC calculates a cryptographic **MD5 hash** of the dataset file.
  - DVC stores the actual file in a content-addressable storage cache or remote repository.
  - DVC generates tiny metadata pointer files (`<filename>.dvc`) containing the MD5 checksum and file size.
  - Git tracks only the lightweight pointer file (`.dvc`) and updates `.gitignore` to prevent raw data from entering Git history.

---

## 2. Content-Addressable Storage (CAS) Mechanics

When you run:
```cmd
dvc add data\raw\dataset_v1.csv
```
DVC performs the following internal operations:
1. Calculates the MD5 checksum of `data\raw\dataset_v1.csv` (e.g., `a7b9c4...`).
2. Takes the first 2 characters of the hash as a subfolder name and the remaining 30 characters as the filename:
   ```
   .dvc/cache/files/md5/a7/b9c4...
   ```
3. Copies or creates a link (reflink, hardlink, or copy depending on OS/filesystem) of the data file into the local `.dvc/cache/`.
4. Creates a pointer file `data\raw\dataset_v1.csv.dvc`:
   ```yaml
   outs:
   - md5: a7b9c48f219d45e69b039cb012a45d2e
     size: 15420
     hash: md5
     path: dataset_v1.csv
   ```
5. Appends `dataset_v1.csv` to `data\raw\.gitignore` so Git ignores the actual raw CSV.

---

## 3. Remote Storage Architecture

DVC decouples local cache storage from team-shared remote storage. Remote storage can be Amazon S3, Google Cloud Storage, Azure Blob, SSH/SFTP, or a shared local directory (for testing and offline exams).

### Configuring a Local Remote in Windows `cmd.exe`:
```cmd
dvc remote add -d myremote %USERPROFILE%\dvc-remote-storage
```
- `-d`: Sets `myremote` as the default remote.
- The path `%USERPROFILE%\dvc-remote-storage` expands to the user profile directory.

### Pushing and Pulling:
- `dvc push`: Uploads cached files matching tracked `.dvc` files to the configured remote.
- `dvc pull`: Downloads datasets from the remote into your local `.dvc/cache` and links them into your workspace.

---

## 4. The Unified DVC + Git Lifecycle

```
[Raw Data] ---> `dvc add data.csv` ---> Updates `.dvc/cache` + Creates `data.csv.dvc`
                                                |
                                                v
[Remote Storage] <--- `dvc push` <--- [Local DVC Cache]
                                                |
[Git Repository] <--- `git add data.csv.dvc` + `git commit`
```

### Standard Versioning Cycle:
1. **Prepare Data:** Create or update `data/raw/dataset_v1.csv`.
2. **Track in DVC:**
   ```cmd
   dvc add data/raw/dataset_v1.csv
   ```
3. **Commit Metadata in Git:**
   ```cmd
   git add data/raw/dataset_v1.csv.dvc data/raw/.gitignore
   git commit -m "feat: track dataset v1"
   ```
4. **Push Raw Data to Remote Storage:**
   ```cmd
   dvc push
   ```

---

## 5. Dataset Evolution, Diffing, and Restoration

### A. Modifying Data (Version 2)
When dataset rows are augmented or changed:
```cmd
python src/augment_data.py
dvc add data/raw/dataset_v1.csv
```
DVC recalculates the MD5 hash and updates `data/raw/dataset_v1.csv.dvc` with the new checksum.

### B. Comparing Dataset Versions (`dvc diff`)
To see the difference between your current workspace data and the previous Git commit:
```cmd
dvc diff
```
Sample Output:
```
modified:
    data/raw/dataset_v1.csv (size: +1.2kB)
```
Or compare between specific Git commits/tags:
```cmd
dvc diff HEAD~1 HEAD
```

### C. Restoring Historical Versions (`git checkout` + `dvc checkout`)
To travel back in time to an earlier dataset version:
1. Check out the desired Git commit or branch:
   ```cmd
   git checkout <v1-commit-hash> -- data/raw/dataset_v1.csv.dvc
   ```
2. Sync the workspace data with the checked-out `.dvc` pointer:
   ```cmd
   dvc checkout
   ```
3. DVC immediately swaps the CSV in `data/raw/` to the exact content corresponding to the v1 MD5 checksum from the cache, in milliseconds without re-downloading!

import os
import sys
from datetime import datetime, timezone, timedelta
from feast import FeatureStore

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[*] Initializing FeatureStore at: {repo_dir}")
    store = FeatureStore(repo_path=repo_dir)

    # Calculate end timestamp in UTC
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)
    end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"[*] Materializing features into SQLite online store up to {end_date_str} (UTC)...")
    try:
        store.materialize(start_date=start_date, end_date=end_date)
        print("[SUCCESS] Materialization completed successfully via Feast API!")
        print(f"  Online store: {os.path.join(repo_dir, 'data', 'online_store.db')}")
    except Exception as e:
        print(f"[ERROR] Materialization failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

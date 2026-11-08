"""Generate a reproducible, explicitly synthetic household grocery dataset."""
import numpy as np
import pandas as pd
import json
import argparse
from src.utils import DATA_PATH, DATA_DIR, ensure_directories, TARGET

SEED = 42
N_RECORDS = 2000

def generate_dataset(n_records=N_RECORDS, seed=SEED):
    rng = np.random.default_rng(seed)
    size = rng.choice(np.arange(1, 9), n_records, p=[.12, .20, .23, .19, .13, .08, .035, .015])
    income = np.clip(rng.lognormal(np.log(52000), .58, n_records), 8000, 350000)
    trips = np.clip(np.rint(rng.normal(7 + size * .75, 2.4)), 1, 30).astype(int)
    items = np.clip(np.rint(rng.normal(13 + size * 3.1, 6)), 3, 70).astype(int)
    branded = np.clip(rng.beta(2.2, 3.2, n_records) * 100 + (income / 350000) * 15, 5, 85)
    fresh = np.clip(rng.beta(3.1, 2.3, n_records) * 100, 12, 90)
    online = np.clip(rng.beta(1.7, 4.5, n_records) * 100 + (income / 350000) * 12, 0, 85)
    # Prior month is known at prediction time; current expenditure is generated
    # around a household demand baseline and that historical observation.
    baseline = (950 + size * 1150 + income * .012 + trips * 95 + items * 16
                + branded * 7.0 + fresh * 4.5 + online * 2.2)
    prior = np.clip(baseline * rng.normal(1.0, .13, n_records), 700, 140000)
    current = np.clip(.42 * baseline + .58 * prior + rng.normal(0, 850 + baseline * .055, n_records), 500, 150000)
    df = pd.DataFrame({
        "household_size": size,
        "monthly_household_income_inr": np.round(income, 2),
        "shopping_trips_per_month": trips,
        "avg_items_per_trip": items,
        "branded_spending_pct": np.round(branded, 2),
        "fresh_produce_spending_pct": np.round(fresh, 2),
        "online_shopping_pct": np.round(online, 2),
        "previous_month_expenditure_inr": np.round(prior, 2),
        TARGET: np.round(current, 2),
    })
    return df

def main(force=False):
    ensure_directories()
    manifest_path = DATA_DIR / "dataset_manifest.json"
    if DATA_PATH.exists():
        if not manifest_path.exists():
            raise FileExistsError(f"Refusing to replace an existing dataset with unknown provenance: {DATA_PATH}. Move it or use the CSV upload workflow.")
        existing=json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("data_kind")!="synthetic":
            raise FileExistsError("Refusing to overwrite a dataset labeled real with generated synthetic data.")
        if not force:
            print(f"Existing SYNTHETIC dataset preserved at {DATA_PATH}; pass --force to regenerate it.")
            return
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    manifest_path.write_text(json.dumps({"data_kind":"synthetic","source_name":"grocery_data.csv","provenance":"Seeded synthetic demonstration generator (seed 42)","seed":SEED},indent=2),encoding="utf-8")
    print(f"Wrote {len(df):,} synthetic records to {DATA_PATH}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--force",action="store_true",help="replace an existing CSV (never use for a real dataset)")
    main(force=parser.parse_args().force)

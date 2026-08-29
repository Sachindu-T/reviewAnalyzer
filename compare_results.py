import os
import pandas as pd

RESULTS_DIR = "results"
comp_rows = []

for name in ["traditional_metrics.csv", "bert_metrics.csv"]:
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        comp_rows.extend(df.to_dict("records"))

if not comp_rows:
    print("No metrics files found in results/. Run the model scripts first.")
else:
    comp = pd.DataFrame(comp_rows)
    out_path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    comp.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (80/10/10 split, 3-class sentiment)")
    print("=" * 60)
    display = comp[["model", "accuracy", "f1_macro", "f1_weighted"]].copy()
    for c in ["accuracy", "f1_macro", "f1_weighted"]:
        display[c] = display[c].map(lambda v: f"{v*100:.2f}%")
    print(display.to_string(index=False))
    print(f"\nSaved -> {out_path}")

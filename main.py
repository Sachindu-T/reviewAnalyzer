import subprocess, sys, os

STEPS = [
    ("Preparing labeled dataset", "prepare_data.py"),
    ("Training traditional TF-IDF baselines", "Traditional_Models_With_TF-IDF.py"),
    ("Fine-tuning BERT", "bert.py"),
    ("Comparing all models", "compare_results.py"),
]

for label, script in STEPS:
    print(f"\n{'#'*70}\n# {label}: {script}\n{'#'*70}")
    result = subprocess.run([sys.executable, script], cwd=os.getcwd())
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

print("\nPipeline complete. See results/ and bert_results/.")

"""
ONE-COMMAND SETUP. Run this first: python run_pipeline.py
Rebuilds every data file, database, and trained model from scratch, in order.
After this finishes, run: streamlit run app.py
"""
import subprocess
import sys

STEPS = [
    ("01_build_catalog.py", "Building the 107-tool catalog"),
    ("02_sql_market_analysis.py", "Running SQL market analysis + building FTS5 search index"),
    ("03_generate_training_data.py", "Generating ML training data (26 categories)"),
    ("04_train_classifier.py", "Training the ML classifier"),
    ("05_test_recommender.py", "Testing the recommender + logging results to SQL"),
    ("06_export_for_powerbi.py", "Exporting Power BI-ready tables"),
]

print("=" * 60)
print("SETTING UP: AI Tool Intelligence Platform")
print("=" * 60)

for script, description in STEPS:
    print(f"\n--- {description} ({script}) ---")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED at {script}")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout.strip()[-400:])

print("\n" + "=" * 60)
print("ALL DONE. Now run:  streamlit run app.py")
print("=" * 60)

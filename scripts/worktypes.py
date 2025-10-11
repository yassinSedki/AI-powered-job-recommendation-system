import pandas as pd
import os

# Load the dataset using a cross-platform path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
file_path = os.path.join(project_root, "dataset", "complete_dataset10k.csv")

print(f"📁 Looking for dataset at: {file_path}")
print(f"📊 File exists: {os.path.exists(file_path)}\n")

df = pd.read_csv(file_path)

# Check available columns first
print("📋 Columns available in the dataset:\n")
print(df.columns.tolist())
print("\n---------------------------------\n")

# Explicitly check for 'work_type_category' column
if 'work_type_category' in df.columns:
    col = 'work_type_category'
    cleaned = (
        df[col]
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
    )
    unique_values = sorted(cleaned.unique())
    print(f"🎯 Unique values in '{col}':\n")
    for val in unique_values:
        print(f"- {val}")
    print("\n📈 Value counts:\n")
    for val, cnt in cleaned.value_counts().items():
        print(f"- {val}: {cnt}")
else:
    print("ℹ️ Column 'work_type_category' not found, trying automatic detection...\n")

# Try to detect a column that might contain work type info
possible_columns = [col for col in df.columns if "work" in col.lower() or "type" in col.lower() or "employment" in col.lower()]
print(f"🕵️ Possible columns related to work type: {possible_columns}\n")

# Prefer known column names if present
known_candidates = ["work_type", "Work Type", "Employment Type", "Job Type", "Employment"]
work_type_column = next((col for col in known_candidates if col in df.columns), None)
if not work_type_column:
    work_type_column = possible_columns[0] if possible_columns else None

if work_type_column:
    unique_values = (
        df[work_type_column]
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
        .unique()
    )
    print(f"🧩 Unique work types found in '{work_type_column}':\n")
    for val in sorted(unique_values):
        print(f"- {val}")
else:
    print("⚠️ No column found that seems to represent 'work type'. Check your dataset column names.")

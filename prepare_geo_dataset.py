import os
import zipfile
import pandas as pd

# --- 1. Unzip dataset ---
zip_path = "archive.zip"
extract_path = "geoguessr_data"

if not os.path.exists(extract_path):
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Done unzipping.")
else:
    print("Already unzipped.")

# --- 2. Load metadata ---
csv_path = os.path.join(extract_path, "locations.csv")
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} entries.")
print(df.head())

# --- 3. Handle small countries ---
# Count images per country
country_counts = df["country"].value_counts()

# Identify small-sample countries
small_countries = country_counts[country_counts < 100].index
print(f"Found {len(small_countries)} countries with <100 samples.")

# Define a function to group them
def group_small_countries(row):
    if row["country"] in small_countries:
        return f"Other ({row['continent']})"
    return row["country"]

df["country_grouped"] = df.apply(group_small_countries, axis=1)

# --- 4. Save cleaned dataset ---
output_csv = os.path.join(extract_path, "geo_dataset_clean.csv")
df.to_csv(output_csv, index=False)
print(f"✅ Cleaned dataset saved to: {output_csv}")

# --- 5. Quick summary ---
print("\nGrouped country counts:")
print(df["country_grouped"].value_counts().head(20))

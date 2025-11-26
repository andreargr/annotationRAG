import pandas as pd
import numpy as np

# 1️⃣ Load the full dataset
df = pd.read_csv("../5K/all_df_5_mini_inference_index.csv", header=0)  # Change filename if needed

# 2️⃣ Define the number of samples per class/type
sample_sizes = {
    "CL": 50,   # Cell lines
    "CT": 38,   # Cell types
    "A": 12     # Anatomical structures
}

# 3️⃣ Set random seed for reproducibility
seed = 42
np.random.seed(seed)

# 4️⃣ Draw stratified samples
samples = []
for label_type, n in sample_sizes.items():
    subset = df[df["Type"] == label_type]
    if len(subset) < n:
        raise ValueError(f"Not enough rows for type '{label_type}' ({len(subset)} < {n})")
    samples.append(subset.sample(n=n, random_state=seed))

# 5️⃣ Combine all samples and shuffle
final_sample = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)

# 6️⃣ Optionally add a unique identifier for traceability
final_sample.insert(0, "Sample_ID", [f"S{i:03d}" for i in range(1, len(final_sample)+1)])

# 7️⃣ Save to CSV
final_sample.to_csv("sample_100_gpt5_inf_index.csv", index=False)

# 8️⃣ Verify results
print("✅ Sample successfully created!")
print("Counts per type:")
print(final_sample["Type"].value_counts())

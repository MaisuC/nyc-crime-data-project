import pandas as pd

# Read your CSV
df = pd.read_csv("nypd_clean.csv")

# Randomly sample rows
df_sampled = df.sample(n=10000, random_state=42)  # set random_state for reproducibility

# Save the result
df_sampled.to_csv("nypd_sampled.csv", index=False)

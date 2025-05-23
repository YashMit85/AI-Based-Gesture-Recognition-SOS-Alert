import pandas as pd

# Load gesture data
wave_data = pd.read_csv("wave_data.csv", header=None)
fist_data = pd.read_csv("fist_data.csv", header=None)

# Add labels (0 = Fist, 1 = Wave)
wave_data["label"] = 1
fist_data["label"] = 0

# Combine both datasets
dataset = pd.concat([wave_data, fist_data], ignore_index=True)

# Save processed dataset
dataset.to_csv("gesture_dataset.csv", index=False)

print("✅ Data processed and saved as gesture_dataset.csv")

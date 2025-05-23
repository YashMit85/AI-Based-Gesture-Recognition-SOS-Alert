import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load the dataset
dataset = pd.read_csv("gesture_dataset.csv")

# Separate features (X) and labels (y)
X = dataset.iloc[:, :-1].values  # All columns except the last (landmarks)
y = dataset.iloc[:, -1].values   # Last column (labels)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

# Save trained model
joblib.dump(svm_model, "gesture_svm.pkl")

print("✅ SVM model trained and saved as gesture_svm.pkl")

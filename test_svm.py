fimport pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("gesture_dataset.csv")

# Separate features (X) and labels (y)
X = dataset.iloc[:, :-1].values  # All columns except the last (landmarks)
y = dataset.iloc[:, -1].values   # Last column (labels)

# Load trained model
svm_model = joblib.load("gesture_svm.pkl")

# Make predictions
y_pred = svm_model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

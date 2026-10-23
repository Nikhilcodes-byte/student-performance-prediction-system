
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# Selected features from our experiment
features = ["17", "29", "30"]
X = df[features]
y = df["GRADE"]

# Keep the same training/test split as the evaluation script
train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = X.loc[train_indices]
y_train = y.loc[train_indices]

# Train the selected model on training records only
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/student_performance_model.joblib")

# Save metadata describing the model
metadata = {
    "model": "Decision Tree Classifier",
    "features": features,
    "target": "GRADE",
    "training_records": len(X_train),
    "test_records_reserved": len(test_indices),
    "random_state": 42,
    "test_size": 0.20
}

with open("models/model_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Model saved successfully.")
print("Model path: models/student_performance_model.joblib")
print("Metadata path: models/model_metadata.json")
print("Training records:", len(X_train))
print("Reserved test records:", len(test_indices))
print("Selected features:", features)


import pandas as pd
import joblib

# Load the saved model
model = joblib.load("models/student_performance_model.joblib")

# Load metadata
metadata = pd.read_json("models/model_metadata.json", typ="series")

features = metadata["features"]

# Create sample inputs using the same feature order
sample = pd.DataFrame(
    [[3, 4, 3]],
    columns=features
)

# Generate a prediction
prediction = model.predict(sample)

print("Saved model loaded successfully.")
print("Input features:", features)
print("Sample input:", sample.iloc[0].to_dict())
print("Predicted grade:", int(prediction[0]))

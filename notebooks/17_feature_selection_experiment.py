
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")
y = df["GRADE"]

# Compare different feature sets
feature_sets = {
    "Set A - 8 features": [
        "17", "20", "22", "25", "26", "27", "29", "30"
    ],
    "Set B - 5 features": [
        "17", "22", "25", "29", "30"
    ],
    "Set C - 3 features": [
        "17", "29", "30"
    ]
}

# Use identical cross-validation folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Feature Selection Experiment")
print("----------------------------")

for name, features in feature_sets.items():
    X = df[features]

    model = DecisionTreeClassifier(random_state=42)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted"
        }
    )

    print(f"\n{name}")
    print("Selected columns:", features)
    print(f"Mean accuracy: {scores['test_accuracy'].mean() * 100:.2f}%")
    print(f"Mean macro F1: {scores['test_f1_macro'].mean():.3f}")
    print(f"Mean weighted F1: {scores['test_f1_weighted'].mean():.3f}")

print("\nExperiment completed.")

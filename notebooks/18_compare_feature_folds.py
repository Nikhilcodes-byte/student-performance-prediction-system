
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("dataset/student_performance.csv")
y = df["GRADE"]

feature_sets = {
    "Set A - 8 features": [
        "17", "20", "22", "25", "26", "27", "29", "30"
    ],
    "Set C - 3 features": [
        "17", "29", "30"
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Fold-by-Fold Feature Comparison")
print("--------------------------------")

for name, features in feature_sets.items():
    X = df[features]
    model = DecisionTreeClassifier(random_state=42)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"\n{name}")
    print("Fold accuracies (%):", (scores * 100).round(2))
    print(f"Mean accuracy: {scores.mean() * 100:.2f}%")
    print(f"Standard deviation: {scores.std() * 100:.2f}%")

print("\nExperiment completed.")

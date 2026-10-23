
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("dataset/student_performance.csv")

features = ["17", "20", "22", "25", "26", "27", "29", "30"]
X = df[features]
y = df["GRADE"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Default Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Balanced Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )
}

print("Class Weight Experiment")
print("-----------------------")

for name, model in models.items():
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
    print(f"Mean accuracy: {scores['test_accuracy'].mean() * 100:.2f}%")
    print(f"Mean macro F1: {scores['test_f1_macro'].mean():.3f}")
    print(f"Mean weighted F1: {scores['test_f1_weighted'].mean():.3f}")

print("\nExperiment completed.")

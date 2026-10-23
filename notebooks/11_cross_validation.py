
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("dataset/student_performance.csv")

# Select the same 8 features used in our earlier experiments
features = ["17", "20", "22", "25", "26", "27", "29", "30"]

X = df[features]
y = df["GRADE"]

# Create 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Define the models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Compare the models
print("5-Fold Cross-Validation Results")
print("--------------------------------")

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"\nModel: {name}")
    print("Fold accuracies (%):", (scores * 100).round(2))
    print(f"Mean accuracy: {scores.mean() * 100:.2f}%")
    print(f"Standard deviation: {scores.std() * 100:.2f}%")

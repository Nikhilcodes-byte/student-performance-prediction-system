
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# Select the same eight features
features = ["17", "20", "22", "25", "26", "27", "29", "30"]
X = df[features]
y = df["GRADE"]

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Evaluate each model
for name, model in models.items():
    y_pred = cross_val_predict(model, X, y, cv=cv)

    print("\n" + "=" * 50)
    print(f"Model: {name}")
    print("=" * 50)

    print(f"Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=list(range(8)),
            zero_division=0
        )
    )

    # Display the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        labels=list(range(8)),
        display_labels=list(range(8)),
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{name} - 5-Fold Cross-Validation")
    plt.tight_layout()
    plt.savefig(
        f"notebooks/{name.lower().replace(' ', '_')}_cv_confusion_matrix.png"
    )
    plt.close()

print("\nAnalysis completed. Confusion matrix images saved.")

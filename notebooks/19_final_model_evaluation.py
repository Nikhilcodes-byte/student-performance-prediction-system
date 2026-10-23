
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# Target
y = df["GRADE"]

# Feature sets
feature_sets = {
    "8-feature model": ["17", "20", "22", "25", "26", "27", "29", "30"],
    "3-feature model": ["17", "29", "30"]
}

# Reserve the final test set
train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

# Five-fold cross-validation on training data only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

print("Final Model Evaluation")
print("======================")
print("Training records:", len(train_indices))
print("Test records:", len(test_indices))

# Compare feature sets using training data only
for name, features in feature_sets.items():
    X_train = df.loc[train_indices, features]
    X_test = df.loc[test_indices, features]

    model = DecisionTreeClassifier(random_state=42)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    results[name] = {
        "features": features,
        "cv_accuracy": scores.mean(),
        "cv_std": scores.std()
    }

    print(f"\n{name}")
    print("Features:", features)
    print("Fold accuracies (%):", (scores * 100).round(2))
    print(f"Mean CV accuracy: {scores.mean() * 100:.2f}%")
    print(f"CV standard deviation: {scores.std() * 100:.2f}%")

# Select feature set based on mean CV accuracy
selected_name = max(
    results,
    key=lambda name: results[name]["cv_accuracy"]
)

selected_features = results[selected_name]["features"]

print("\nSelected candidate:", selected_name)
print("Selected features:", selected_features)

# Fit selected model on all training data
final_model = DecisionTreeClassifier(random_state=42)
final_model.fit(df.loc[train_indices, selected_features], y_train)

# Evaluate once on the untouched test set
y_pred = final_model.predict(df.loc[test_indices, selected_features])

print("\nUntouched Test Set Results")
print("--------------------------")
print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(8)),
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=list(range(8))))

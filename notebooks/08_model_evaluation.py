import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# 2. Select features
features = [
    "17",  # Weekly study hours
    "20",  # Seminar/conference attendance
    "22",  # Class attendance
    "25",  # Note-taking
    "26",  # Listening in class
    "27",  # Discussion
    "29",  # Previous GPA
    "30"   # Expected graduation GPA
]

X = df[features]
y = df["GRADE"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Create models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# 5. Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n", name)
    print("-" * len(name))
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Display confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()
    plt.title(name)
    plt.savefig(
        f"notebooks/{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.close()

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# Selected features and target
features = ["17", "20", "22", "25", "26", "27", "29", "30"]
X = df[features]
y = df["GRADE"]

# Feature groups
ordinal_features = ["17", "22", "25", "26", "27", "29", "30"]
binary_features = ["20"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "ordinal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler())
            ]),
            ordinal_features
        ),
        (
            "binary",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            binary_features
        )
    ]
)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Same cross-validation folds for every model
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Preprocessing Experiment")
print("------------------------")

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted"
        }
    )

    print(f"\nModel: {name}")
    print(f"Mean accuracy: {scores['test_accuracy'].mean() * 100:.2f}%")
    print(f"Mean precision: {scores['test_precision'].mean():.3f}")
    print(f"Mean recall: {scores['test_recall'].mean():.3f}")
    print(f"Mean F1-score: {scores['test_f1'].mean():.3f}")
    print(
        "Accuracy standard deviation: "
        f"{scores['test_accuracy'].std() * 100:.2f}%"
    )

print("\nExperiment completed.")

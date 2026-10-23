import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# 3. Separate input and target
X = df[features]
y = df["GRADE"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# 6. Train
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Model")
print("-------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
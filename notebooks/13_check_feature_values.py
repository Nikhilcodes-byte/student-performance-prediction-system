
import pandas as pd

df = pd.read_csv("dataset/student_performance.csv")

features = {
    "17": "Weekly Study Hours",
    "20": "Seminar Attendance",
    "22": "Class Attendance",
    "29": "Previous GPA",
    "30": "Expected Graduation GPA"
}

for column, name in features.items():
    print(f"\n{name} (Column {column})")
    print("Unique values:", sorted(df[column].unique()))
    print("Value counts:")
    print(df[column].value_counts().sort_index())


import pandas as pd

df = pd.read_csv("dataset/student_performance.csv")

print("Grade distribution")
print("------------------")

counts = df["GRADE"].value_counts().sort_index()

for grade, count in counts.items():
    percentage = count / len(df) * 100
    print(
        f"Grade {grade}: {count} students "
        f"({percentage:.2f}%)"
    )

print("\nTotal records:", len(df))

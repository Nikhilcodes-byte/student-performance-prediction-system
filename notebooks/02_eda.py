import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# -----------------------------
# 1. Basic statistics
# -----------------------------

print("BASIC STATISTICS")
print(df.describe())

# -----------------------------
# 2. Grade distribution
# -----------------------------

print("\nGRADE DISTRIBUTION")
print(df["GRADE"].value_counts().sort_index())

plt.figure(figsize=(8, 5))
sns.countplot(x="GRADE", data=df)
plt.title("Student Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("notebooks/grade_distribution.png")
plt.close()

# -----------------------------
# 3. Previous GPA vs Grade
# -----------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(x="GRADE", y="29", data=df)
plt.title("Previous GPA Feature vs Grade")
plt.xlabel("Grade")
plt.ylabel("Previous GPA Category")
plt.tight_layout()
plt.savefig("notebooks/previous_gpa_vs_grade.png")
plt.close()

# -----------------------------
# 4. Expected GPA vs Grade
# -----------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(x="GRADE", y="30", data=df)
plt.title("Expected Graduation GPA vs Grade")
plt.xlabel("Grade")
plt.ylabel("Expected GPA Category")
plt.tight_layout()
plt.savefig("notebooks/expected_gpa_vs_grade.png")
plt.close()

# -----------------------------
# 5. Study Hours vs Grade
# -----------------------------

plt.figure(figsize=(8, 5))
sns.countplot(x="17", hue="GRADE", data=df)
plt.title("Weekly Study Hours vs Grade")
plt.xlabel("Weekly Study Hours Category")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("notebooks/study_hours_vs_grade.png")
plt.close()

# -----------------------------
# 6. Class Attendance vs Grade
# -----------------------------

plt.figure(figsize=(8, 5))
sns.countplot(x="22", hue="GRADE", data=df)
plt.title("Class Attendance vs Grade")
plt.xlabel("Class Attendance Category")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("notebooks/attendance_vs_grade.png")
plt.close()

print("\nEDA completed successfully.")
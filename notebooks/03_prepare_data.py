import pandas as pd

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# Features selected for the first ML experiment
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

# Input features
X = df[features]

# Target
y = df["GRADE"]

print("FEATURES (X)")
print(X.head())

print("\nTARGET (y)")
print(y.head())

print("\nX SHAPE")
print(X.shape)

print("\nY SHAPE")
print(y.shape)

print("\nFEATURE NAMES")
print(features)
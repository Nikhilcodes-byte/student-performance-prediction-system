import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/student_performance.csv")

# Display first 5 rows
print("FIRST 5 ROWS")
print(df.head())

# Display number of rows and columns
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Display data types
print("\nDATA TYPES")
print(df.dtypes)

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Display basic information
print("\nDATASET INFORMATION")
df.info()

# Display target distribution
print("\nOUTPUT GRADE DISTRIBUTION")
print(df["GRADE"].value_counts().sort_index())
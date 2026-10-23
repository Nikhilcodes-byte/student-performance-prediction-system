import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("dataset/student_performance.csv")

# 2. Remove ID columns
df_features = df.drop(columns=["STUDENT ID", "COURSE ID"])

# 3. Calculate correlation with GRADE
correlations = df_features.corr()["GRADE"].sort_values(ascending=False)

print("Correlation with GRADE")
print("----------------------")
print(correlations)

# 4. Plot correlations
plt.figure(figsize=(10, 8))

correlations.drop("GRADE").sort_values().plot(kind="barh")

plt.title("Feature Correlation with Student Grade")
plt.xlabel("Correlation")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig("notebooks/feature_correlation.png")

plt.close()

print("\nCorrelation graph saved successfully.")
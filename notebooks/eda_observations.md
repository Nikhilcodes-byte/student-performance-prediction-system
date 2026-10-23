# Exploratory Data Analysis Observations

## 1. Grade Distribution

The dataset contains eight grade categories ranging from 0 to 7.
Grade 1 is the most frequent category with 35 observations,
while Grade 0 has the fewest observations with 8 observations.

The target classes are therefore not perfectly balanced.

## 2. Previous GPA vs Grade

The analysis indicates that higher performance grades generally
contain higher previous GPA categories than several lower-grade
groups. However, there is considerable overlap between some
grade categories.

Previous GPA appears to be a potentially useful predictive feature.

## 3. Expected Graduation GPA vs Grade

The expected graduation GPA feature shows variation across
different grade categories. Higher grade groups generally contain
higher expected-GPA categories, although overlap exists.

This feature may provide useful information for prediction.

## 4. Weekly Study Hours vs Grade

Most students belong to the lower or middle study-hour categories.
Different performance grades are present across several study-hour
categories.

Study hours may therefore provide useful predictive information,
but the relationship is not perfectly linear.

## 5. Class Attendance vs Grade

Most observations belong to the first attendance category.
Both attendance categories contain students from multiple
performance-grade groups.

Class attendance may provide useful information to the model,
although the graph does not indicate a simple direct relationship.

## Overall Observation

The EDA suggests that several educational and behavioural
features may contain useful information for predicting student
performance. Further feature analysis and model-based evaluation
will be performed before finalising the input features.
# Student Performance Prediction System
## Machine Learning Plan

## 1. Problem Type

The project will be treated as a supervised machine learning
classification problem.

The objective is to predict the student's academic performance
grade using selected student-related features.

## 2. Input Features

The initial feature set will contain:

- Feature 17: Weekly study hours
- Feature 20: Seminar/conference attendance
- Feature 22: Class attendance
- Feature 25: Note-taking
- Feature 26: Listening in class
- Feature 27: Discussion
- Feature 29: Previous GPA
- Feature 30: Expected graduation GPA

## 3. Target Variable

The target variable is:

GRADE

The target contains performance categories from 0 to 7.

## 4. Data Preparation

The following steps will be performed:

1. Load the dataset.
2. Select the required input features.
3. Separate input features and target variable.
4. Divide the data into training and testing sets.
5. Prepare the data in a suitable format for machine learning.

## 5. Classification Algorithms

The initial candidate algorithms will be:

1. Decision Tree Classifier
2. Logistic Regression
3. Random Forest Classifier

The algorithms will be trained and evaluated using the same
dataset split for comparison.

## 6. Model Evaluation

The candidate models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The final model will be selected based on the observed
evaluation results and suitability for the project.

## 7. Model Integration

After selecting the final model:

1. The trained model will be saved.
2. Flask will load the saved model.
3. User input from the web form will be passed to the model.
4. The model will generate a predicted grade.
5. The prediction will be displayed to the user.
6. The relevant prediction record will be stored in MySQL.
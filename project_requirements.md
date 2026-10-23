# Student Performance Prediction System

## 1. Project Overview

The Student Performance Prediction System is a web-based machine learning application that predicts a student's academic performance based on selected educational and behavioural attributes.

The system will use supervised machine learning classification algorithms and will provide predictions through a Flask-based web application.

CSV files will be used to store prediction history, allowing users to review previous predictions without requiring a separate database server.

---

## 2. Project Objective

The main objective of this project is to develop a system that
can process student-related information and predict the student's
academic performance using a machine learning classification model.

The system will also provide a simple web interface for entering
student information and viewing prediction results.

---

## 3. Functional Requirements

The system should:

1. Accept student information through a web interface.

2. Validate the information entered by the user.

3. Preprocess the input data before sending it to the machine
   learning model.

4. Predict the student's academic performance.

5. Display the prediction result clearly.

6. Store relevant prediction records in a MySQL database.

7. Retrieve previously stored prediction records.

8. Provide basic performance analysis and visualisation.

9. Allow testing of the complete prediction workflow.

---

## 4. Machine Learning Requirements

The machine learning component should:

1. Use a suitable student performance dataset.

2. Perform data preprocessing.

3. Perform exploratory data analysis.

4. Select relevant features.

5. Train multiple suitable classification algorithms.

6. Evaluate the trained models using appropriate evaluation metrics.

7. Compare the performance of the candidate models.

8. Select a suitable model for the final application.

9. Save the selected trained model for use by the Flask application.

---

## 5. Web Application Requirements

The web application should contain:

1. A home page.

2. A student information input form.

3. A prediction functionality.

4. A prediction result page.

5. A dashboard or basic analysis section.

6. A section for viewing previous prediction records.

---

## 6. Database Requirements

MySQL will be used as the relational database.

The database should store relevant:

- Student information
- Prediction records
- Prediction results
- Required timestamps or record information

The database structure will be finalised during the
system-design phase.

---

## 7. Technology Requirements

### Programming Language
Python

### Machine Learning
Scikit-learn

### Data Processing
Pandas and NumPy

### Backend
Flask

### Frontend
HTML, CSS and JavaScript

### Database
MySQL

### Visualisation
Matplotlib and Seaborn

### Development Environment
Visual Studio Code

### Version Control
Git and GitHub

---

## 8. Scope

The project will focus on developing an academic prototype
for student performance prediction.

The system will use the selected public dataset and the
features available in that dataset.

The project will include machine learning, web application
development and database integration.

The system will not replace formal academic assessment or
teacher judgement.

---

## 9. Expected Users

The intended users of the prototype are:

- Students
- Faculty members
- Academic project demonstrators

---

## 10. Project Limitations

The prediction quality will depend on the dataset and the
selected machine learning model.

The system will be treated as an academic prototype and
its results will not be presented as universally accurate.

A larger and more diverse dataset could be used in future
versions of the system.
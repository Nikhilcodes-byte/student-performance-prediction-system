# Student Performance Prediction System
## System Architecture

The Student Performance Prediction System will use a
three-layer web application architecture consisting of the
presentation layer, application layer and data/model layer.

## 1. Presentation Layer

The presentation layer will be developed using HTML, CSS
and JavaScript.

It will provide:

- Home page
- Student information form
- Prediction result
- Basic dashboard
- Prediction history

The user will enter the required student information through
the web interface.

## 2. Application Layer

Flask will be used as the backend framework.

The application layer will:

- Receive user input
- Validate the input
- Process the input data
- Pass the processed data to the machine learning model
- Receive the prediction
- Display the prediction result
- Communicate with the MySQL database

## 3. Machine Learning Layer

The machine learning component will be developed using
Python and Scikit-learn.

It will include:

- Dataset preprocessing
- Feature selection
- Model training
- Model evaluation
- Model comparison
- Final model selection
- Student performance prediction

The trained model will be saved and loaded by the Flask
application when predictions are required.

## 4. Database Layer

MySQL will be used as the relational database.

The database will store relevant:

- Student information
- Prediction results
- Prediction records
- Date/time information where required

Flask will communicate with MySQL to store and retrieve
application records.

## 5. Overall Data Flow

The overall system will work as follows:

1. The user opens the web application.
2. The user enters the required student information.
3. Flask receives and validates the submitted information.
4. The input is converted into the format required by the
   machine learning model.
5. The trained model generates a predicted performance grade.
6. The prediction result is displayed to the user.
7. Relevant information and the prediction result are stored
   in MySQL.
8. Stored prediction records can be retrieved for the
   prediction history/dashboard.

## 6. Technologies

- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- Machine Learning: Scikit-learn
- Data Processing: Pandas, NumPy
- Database: MySQL
- Visualisation: Matplotlib, Seaborn
- Development Environment: Visual Studio Code
- Version Control: Git and GitHub
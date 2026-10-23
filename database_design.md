# Student Performance Prediction System
## Database Design

## Database Name

student_performance_db

## 1. Students Table

The students table stores the student information entered
through the web application.

| Column | Data Type | Description |
|---|---|---|
| student_id | INT | Primary key and unique student identifier |
| student_name | VARCHAR(100) | Name of the student |
| study_hours | INT | Weekly study-hours category |
| seminar_attendance | INT | Seminar/conference attendance category |
| class_attendance | INT | Class attendance category |
| note_taking | INT | Note-taking category |
| listening | INT | Listening-in-class category |
| discussion | INT | Discussion category |
| previous_gpa | INT | Previous GPA category |
| expected_gpa | INT | Expected graduation GPA category |

## 2. Predictions Table

The predictions table stores prediction results generated
by the machine learning model.

| Column | Data Type | Description |
|---|---|---|
| prediction_id | INT | Primary key |
| student_id | INT | Foreign key referencing students |
| predicted_grade | INT | Predicted performance grade |
| prediction_date | DATETIME | Date and time of prediction |

## 3. Relationship

The `student_id` column in the predictions table references
the `student_id` column in the students table.

One student can have multiple prediction records.

## 4. Purpose

The database will allow the application to:

- Store student information.
- Store prediction results.
- Maintain prediction history.
- Retrieve previous prediction records.
- Support basic dashboard information.

## 5. Database Technology

MySQL will be used as the relational database management
system.
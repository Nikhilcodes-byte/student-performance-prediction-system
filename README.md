# Student Performance Prediction System

A machine learning-based web application that predicts student performance grades from selected academic attributes. Built with Python, Flask, and Scikit-learn, the application includes prediction history and a dashboard.

## Features

- Predict a student's grade using selected academic attributes
- Web interface built with HTML, CSS, and JavaScript
- Machine learning model integrated with Flask
- Store prediction history in a CSV file
- View previous predictions in a history table
- Dashboard showing prediction counts, grade distribution, and recent predictions
- Data analysis and model evaluation experiments

## Tech Stack

- **Language:** Python
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Data Storage:** CSV
- **Data Visualization:** Matplotlib, Seaborn
- **Version Control:** Git and GitHub

## Project Structure

```text
Student-Performance-Prediction/
├── app/
│   └── app.py
├── dataset/
│   ├── student_performance.csv
│   └── dataset_dictionary.md
├── models/
│   ├── student_performance_model.joblib
│   └── model_metadata.json
├── notebooks/
├── static/
├── templates/
│   ├── index.html
│   ├── history.html
│   └── dashboard.html
├── .gitignore
├── requirements.txt
├── README.md
├── database_design.md
├── ml_plan.md
├── project_requirements.md
├── project_scope.md
├── system_design.md
└── ui_design.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nikhilcodes-byte/student-performance-prediction-system.git
cd student-performance-prediction-system
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app/app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

## Prediction History and Dashboard

The application stores prediction records in `prediction_history.csv`. This file is created automatically when the first prediction is saved.

- **Prediction page:** Enter the required academic attributes and generate a prediction.
- **Prediction History:** View previously saved predictions.
- **Dashboard:** View summary statistics and the distribution of predicted grades.

Prediction history is stored locally and is excluded from version control by `.gitignore`.

## Dataset

The project uses the [UCI Higher Education Students Performance Evaluation dataset](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation).

Refer to `dataset/dataset_dictionary.md` for the feature descriptions used in the project.

## Model Evaluation and Limitations

Several classification algorithms and feature combinations were explored during development. The final experimental model uses a Decision Tree classifier with selected features.

In the recorded evaluation, the model achieved **24.14% accuracy on a held-out test set of 29 records**. The dataset is small and the model's predictive performance is limited.

Predictions are experimental estimates, not guaranteed academic outcomes. The application should not be used to make important academic decisions.

See the `notebooks/` directory for data analysis, experiments, and evaluation results.

## Future Improvements

- Improve model performance through further experimentation and validation
- Enhance the dashboard and data visualizations
- Add automated testing and improved error handling
- Deploy the application for public demonstration

## Author

**Nikhilcodes-byte**

GitHub: [@Nikhilcodes-byte](https://github.com/Nikhilcodes-byte)
import csv
from datetime import datetime

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from flask import Flask, jsonify, request, render_template

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"

app = Flask(
    __name__,
    template_folder="../templates"
)

# Load the trained model once when the application starts
model = joblib.load(MODEL_PATH)

# The exact feature order used during model training
FEATURES = ["17", "29", "30"]
HISTORY_PATH = PROJECT_ROOT / "prediction_history.csv"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    try:
        values = [float(data[feature]) for feature in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric values for features 17, 29, and 30."
        }), 400

    # Validate the encoded category ranges
    allowed_values = {
        "17": [1, 2, 3, 4, 5],
        "29": [1, 2, 3, 4, 5],
        "30": [1, 2, 3, 4]
    }

    for feature, value in zip(FEATURES, values):
        if not value.is_integer() or int(value) not in allowed_values[feature]:
            return jsonify({
                "error": f"Feature {feature} has an invalid category value."
            }), 400

    input_data = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(input_data)[0])

    grade_labels = {
        0: "Fail",
        1: "DD",
        2: "DC",
        3: "CC",
        4: "CB",
        5: "BB",
        6: "BA",
        7: "AA"
    }

    # Save prediction history to CSV
    file_exists = HISTORY_PATH.exists()

    with open(HISTORY_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Add column headings only when creating the file
        if not file_exists:
            writer.writerow([
                "timestamp",
                "weekly_study_hours",
                "previous_gpa",
                "expected_gpa",
                "predicted_grade",
                "grade_label"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(values[0]),
            int(values[1]),
            int(values[2]),
            prediction,
            grade_labels.get(prediction, "Unknown")
        ])

    return jsonify({
        "predicted_grade": prediction,
        "grade_label": grade_labels.get(prediction, "Unknown"),
        "notice": "This is an experimental prediction, not a guaranteed academic outcome."
    })

@app.route("/history")
def history():
    records = []

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            records = list(reader)

    records.reverse()

    return render_template("history.html", records=records)


@app.route("/dashboard")
def dashboard():
    records = []

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            records = list(reader)

    total_predictions = len(records)

    grade_counts = {}
    for record in records:
        grade = record["grade_label"]
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    most_common_grade = (
        max(grade_counts, key=grade_counts.get)
        if grade_counts else "N/A"
    )

    recent_records = list(reversed(records[-5:]))

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        grade_counts=grade_counts,
        most_common_grade=most_common_grade,
        recent_records=recent_records
    )


import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )


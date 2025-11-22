from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & dataset
model = joblib.load("student_performance_model.pkl")
df = pd.read_csv("StudentsPerformance.csv")

# Dropdown list
categorical_columns = [
    "gender", "race/ethnicity",
    "parental level of education",
    "lunch", "test preparation course"
]
unique_values = {col: sorted(df[col].unique()) for col in categorical_columns}

# Encoding
encode_gender = {"female": 0, "male": 1}
encode_race = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}

encode_parent = {
    "associate's degree": 0,
    "bachelor's degree": 1,
    "high school": 2,
    "master's degree": 3,
    "some college": 4,
    "some high school": 5
}

encode_lunch = {"free/reduced": 0, "standard": 1}

encode_prep = {"completed": 0, "none": 1}



@app.route("/")
def home():
    return render_template("index.html", data=unique_values)


@app.route("/predict", methods=["POST"])
def predict():

    cols = [
        "gender", "race/ethnicity",
        "parental level of education",
        "lunch", "test preparation course",
        "math score", "reading score", "writing score"
    ]

    user_inputs = []

    # Ambil input user
    for col in cols:
        raw = request.form[col]

        if col == "gender":
            value = encode_gender[raw]
        elif col == "race/ethnicity":
            value = encode_race[raw]
        elif col == "parental level of education":
            value = encode_parent[raw]
        elif col == "lunch":
            value = encode_lunch[raw]
        elif col == "test preparation course":
            value = encode_prep[raw]
        else:
            value = float(raw)

        user_inputs.append(value)

    # Nilai nilai
    math_score = float(request.form["math score"])
    reading_score = float(request.form["reading score"])
    writing_score = float(request.form["writing score"])

    avg_score = round((math_score + reading_score + writing_score) / 3, 2)

    # Prediksi model
    input_df = pd.DataFrame([user_inputs], columns=cols)
    prediction = model.predict(input_df)[0]

    kategori = ["Rendah", "Sedang", "Tinggi"][prediction]
    result = f"Hasil Prediksi: {kategori} (Nilai rata-rata: {avg_score})"

    # Kirim ke HTML untuk Plotly
    graph_labels = ["Math", "Reading", "Writing"]
    graph_values = [math_score, reading_score, writing_score]

    return render_template(
        "index.html",
        data=unique_values,
        prediction_text=result,
        labels=graph_labels,
        values=graph_values
    )


if __name__ == "__main__":
    app.run(debug=True)

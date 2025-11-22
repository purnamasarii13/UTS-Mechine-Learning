from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# ===================== 1. Load & siapkan dataset =====================

df = pd.read_csv("StudentsPerformance.csv")

# fungsi label performa (sama seperti di students.py)
def label_performance(n: float) -> str:
    if n < 60:
        return "rendah"
    elif n < 80:
        return "sedang"
    else:
        return "tinggi"

# kolom rata-rata dan label performa
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["performance"] = df["avg_score"].apply(label_performance)

# ===================== 2. Encoding manual (konsisten) =====================

encode_gender = {"female": 0, "male": 1}

encode_race = {
    "group A": 0,
    "group B": 1,
    "group C": 2,
    "group D": 3,
    "group E": 4,
}

encode_parent = {
    "associate's degree": 0,
    "bachelor's degree": 1,
    "high school": 2,
    "master's degree": 3,
    "some college": 4,
    "some high school": 5,
}

encode_lunch = {"free/reduced": 0, "standard": 1}

encode_prep = {"completed": 0, "none": 1}

encode_target = {"rendah": 0, "sedang": 1, "tinggi": 2}

# buat salinan df untuk di-encode
df_enc = df.copy()
df_enc["gender"] = df_enc["gender"].map(encode_gender)
df_enc["race/ethnicity"] = df_enc["race/ethnicity"].map(encode_race)
df_enc["parental level of education"] = df_enc["parental level of education"].map(encode_parent)
df_enc["lunch"] = df_enc["lunch"].map(encode_lunch)
df_enc["test preparation course"] = df_enc["test preparation course"].map(encode_prep)
df_enc["performance"] = df_enc["performance"].map(encode_target)

# ===================== 3. Train model decision tree =====================

feature_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score",
]

X = df_enc[feature_cols]
y = df_enc["performance"]

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# ===================== 4. Data untuk dropdown di form =====================

categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
unique_values = {col: sorted(df[col].unique()) for col in categorical_columns}

# ===================== 5. Routes Flask =====================

@app.route("/")
def home():
    return render_template("index.html", data=unique_values)


@app.route("/predict", methods=["POST"])
def predict():
    # urutan kolom harus sama dengan saat training
    cols = feature_cols

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
            # kolom nilai numerik
            value = float(raw)

        user_inputs.append(value)

    # ambil nilai untuk ditampilkan di grafik
    math_score = float(request.form["math score"])
    reading_score = float(request.form["reading score"])
    writing_score = float(request.form["writing score"])

    avg_score = round((math_score + reading_score + writing_score) / 3, 2)

    # Prediksi model
    input_df = pd.DataFrame([user_inputs], columns=cols)
    pred_class = model.predict(input_df)[0]

    kategori = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}[pred_class]
    result = f"Hasil Prediksi: {kategori} (Nilai rata-rata: {avg_score})"

    # Kirim ke HTML untuk Plotly
    graph_labels = ["Math", "Reading", "Writing"]
    graph_values = [math_score, reading_score, writing_score]

    return render_template(
        "index.html",
        data=unique_values,
        prediction_text=result,
        labels=graph_labels,
        values=graph_values,
    )


if __name__ == "__main__":
    app.run(debug=True)

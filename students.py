# ============================================================
#           PREDIKSI PERFORMA AKADEMIK SISWA
#          MENGGUNAKAN ALGORITMA DECISION TREE
# ============================================================

# ============== 1. Import Library ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

import joblib  # <-- TAMBAHAN UNTUK SIMPAN MODEL

# ============== 2. Load Dataset =============================
df = pd.read_csv("StudentsPerformance.csv")
print("Dataset Awal:")
print(df.head())

# ============== 3. Membuat Kolom Rata-rata Nilai ============
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

def label_performance(n):
    if n < 60:
        return "rendah"
    elif n < 80:
        return "sedang"
    else:
        return "tinggi"

df["performance"] = df["avg_score"].apply(label_performance)

# ============== 4. Label Encoding untuk Fitur Kategorikal ===
cat_cols = ["gender", "race/ethnicity", "parental level of education",
            "lunch", "test preparation course"]

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# encode target
df["performance"] = le.fit_transform(df["performance"])

print("\nDataset Setelah Encoding:")
print(df.head())

# ============== 5. Pilih Fitur dan Label ====================
X = df[[
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score"
]]

y = df["performance"]

# ============== 6. Train-Test Split =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============== 7. Train Model Decision Tree ================
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ============== 8. Visualisasi Pohon Keputusan ==============
plt.figure(figsize=(25, 12))
plot_tree(
    model,
    feature_names=list(X.columns),   # ← perbaikan penting
    class_names=["Rendah", "Sedang", "Tinggi"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Visualisasi Decision Tree - Prediksi Performa Akademik Siswa")
plt.show()

# ============== 9. Confusion Matrix =========================
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Rendah", "Sedang", "Tinggi"],
    yticklabels=["Rendah", "Sedang", "Tinggi"]
)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

print("\n=== Classification Report ===")
print(classification_report(y_test, predictions))

# ============== 10. Feature Importance ======================
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Pentingnya": importances
}).sort_values(by="Pentingnya", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Pentingnya", y="Fitur", data=feature_importance_df, palette="viridis")
plt.title("Peringkat Pentingnya Fitur")
plt.xlabel("Nilai Importance")
plt.ylabel("Fitur")
plt.show()

# ============== 11. ROC Curve (One-vs-Rest untuk 3 kelas) ===
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_pred_proba = model.predict_proba(X_test)

plt.figure(figsize=(8, 6))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f"Kelas {i} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("ROC Curve (One-vs-Rest) - Decision Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

# ============== 12. Simpan Model ke File PKL ================
joblib.dump(model, "student_performance_model.pkl")
print("Model berhasil disimpan sebagai 'student_performance_model.pkl'")

# ==================== SELESAI ================================

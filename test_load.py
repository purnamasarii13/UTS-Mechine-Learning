import joblib
import pandas as pd

print("Coba load model...")
model = joblib.load("student_performance_model.pkl")
print(" >> BERHASIL load model")

print("Coba load CSV...")
df = pd.read_csv("StudentsPerformance.csv")
print(" >> BERHASIL load CSV")

print("Semua OK")

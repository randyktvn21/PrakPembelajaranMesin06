from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model dan fitur
model = joblib.load("model_random_forest_stroke.joblib")
features = joblib.load("features_stroke.joblib")

def build_input_df(data_dict):
    df = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=features, fill_value=0)
    return df_encoded


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            "gender": request.form.get("gender"),
            "age": float(request.form.get("age")),
            "hypertension": int(request.form.get("hypertension")),
            "heart_disease": int(request.form.get("heart_disease")),
            "ever_married": request.form.get("ever_married"),
            "work_type": request.form.get("work_type"),
            "Residence_type": request.form.get("Residence_type"),
            "avg_glucose_level": float(request.form.get("avg_glucose_level")),
            "bmi": float(request.form.get("bmi")),
            "smoking_status": request.form.get("smoking_status"),
        }

        df_input = build_input_df(data)
        proba = model.predict_proba(df_input)[0, 1]
        pred = model.predict(df_input)[0]

        hasil = "⚠️ Pasien Berisiko Stroke" if pred == 1 else "✅ Risiko Stroke Rendah"
        prob_text = f"{proba*100:.2f}%"

        return render_template('index.html', hasil=hasil, prob=prob_text, data=data)
    except Exception as e:
        return render_template('index.html', hasil=f"Terjadi kesalahan: {e}")


@app.route('/contoh')
def contoh():
    """Contoh data dengan format yang cocok untuk model"""
    contoh_data = {
        "gender": "Female",
        "age": 67,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Govt_job",
        "Residence_type": "Urban",
        "avg_glucose_level": 210.3,
        "bmi": 29.6,
        "smoking_status": "formerly smoked"
    }
    return jsonify(contoh_data)


if __name__ == '__main__':
    app.run(debug=True)

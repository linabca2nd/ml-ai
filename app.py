from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("diabetes_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form

        gender = request.form["gender"]
        age = float(request.form["age"])

        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])

        smoking_history = request.form["smoking_history"]

        bmi = float(request.form["bmi"])
        hba1c = float(request.form["hba1c"])
        glucose = float(request.form["glucose"])

        # Validate values

        if not 0.08 <= age <= 80:
            raise ValueError("Invalid age")

        if not 10 <= bmi <= 100:
            raise ValueError("Invalid BMI")

        if not 3 <= hba1c <= 20:
            raise ValueError("Invalid HbA1c level")

        if not 50 <= glucose <= 500:
            raise ValueError("Invalid glucose level")

        if hypertension not in [0, 1]:
            raise ValueError("Invalid hypertension")

        if heart_disease not in [0, 1]:
            raise ValueError("Invalid heart disease")

        if gender not in ["Female", "Male", "Other"]:
            raise ValueError("Invalid gender")

        if smoking_history not in [
            "never", "No Info", "current",
            "former", "not current", "ever"
        ]:
            raise ValueError("Invalid smoking history")

        # Convert input into DataFrame

        input_data = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose
        }])

        # Predict diabetes

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        probability = round(probability * 100, 2)

        if prediction == 1:
            result = "Higher predicted likelihood of diabetes"
        else:
            result = "Lower predicted likelihood of diabetes"

        return render_template(
            "result.html",
            result=result,
            probability=probability
        )

    except (ValueError, KeyError):
        return render_template(
            "result.html",
            error="Invalid input. Please enter valid values."
        )


if __name__ == "__main__":
    app.run(debug=True)
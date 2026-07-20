"""
Epic 5: Application Building
Story: Building the Python Script, Run the Application
"""

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Story: Building the Python Script - load the trained model saved in Epic 4
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs (map/encode these to match your training feature order)
    gender = request.form.get("gender")
    married = request.form.get("married")
    dependents = request.form.get("dependents")
    education = request.form.get("education")
    self_employed = request.form.get("self_employed")
    applicant_income = float(request.form.get("applicant_income", 0))
    coapplicant_income = float(request.form.get("coapplicant_income", 0))
    loan_amount = float(request.form.get("loan_amount", 0))
    loan_term = float(request.form.get("loan_term", 360))
    credit_history = float(request.form.get("credit_history", 1))
    property_area = request.form.get("property_area")

    # TODO: apply the same encoding/scaling used during training (Epic 3)
    features = np.array([[applicant_income, coapplicant_income, loan_amount,
                           loan_term, credit_history]])

    prediction = model.predict(features)[0]
    result_text = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

    return render_template("result.html", prediction_text=result_text)


if __name__ == "__main__":
    app.run(debug=True)

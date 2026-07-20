import os
import pickle
import numpy as np
import pandas as pd

from flask import Flask, render_template, request

# ==========================================================
# CREATE FLASK APPLICATION
# ==========================================================

app = Flask(__name__)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = pickle.load(open("rdf.pkl", "rb"))

# ==========================================================
# LOAD SCALER
# ==========================================================

scaler = pickle.load(open("scale1.pkl", "rb"))

# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():
    return render_template("home.html")


# ==========================================================
# PREDICTION PAGE
# ==========================================================

@app.route("/predict")
def predict():
    return render_template("predict.html")


# ==========================================================
# SUBMIT
# ==========================================================

@app.route("/submit", methods=["POST"])
def submit():

    try:

        input_features = [float(x) for x in request.form.values()]

        columns = [

            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area"

        ]

        data = pd.DataFrame([input_features], columns=columns)

        # Scale input
        data = scaler.transform(data)

        # Predict
        prediction = model.predict(data)

        prediction = int(prediction[0])

        if prediction == 1:

            result = "Loan Approved"

        else:

            result = "Loan Rejected"

        return render_template(
            "output.html",
            result=result
        )

    except Exception as e:

        return render_template(
            "output.html",
            result=f"Error : {e}"
        )


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":

    app.run(debug=True)
# Epic 1: Data Collection and Architecture Design

## Story: Download the Dataset

- Dataset source: _add link here (e.g. Kaggle Loan Prediction Dataset)_
- Save the raw dataset as `data/loan_data.csv`
- Columns typically include: `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status`

## Story: Defining the Application Architecture

See [Project Design Phase](../../3.%20Project%20Design%20Phase/README.md) for the full architecture diagram.

Summary:
- **Frontend:** HTML form for user input
- **Backend:** Flask app (`app.py`) loads the trained model and serves predictions
- **Model:** Saved as a `.pkl` file after training in Epic 4

# Smart Lender - Loan Approval Prediction using Machine Learning

## Project Overview

Smart Lender is a Machine Learning-based web application that predicts whether a loan application is likely to be approved or rejected based on applicant information.

The project is designed to assist banks and financial institutions in making faster and more accurate loan approval decisions using historical loan application data.

The web application is developed using Python, Flask, HTML, CSS, and Scikit-Learn.

---

## Features

- Loan Approval Prediction using Machine Learning
- User-Friendly Web Interface
- Real-Time Prediction
- Data Preprocessing
- Feature Scaling
- Multiple Machine Learning Algorithms
- Responsive Design

---

## Technologies Used

### Programming Language

- Python

### Frontend

- HTML5
- CSS3

### Backend

- Flask

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Model Serialization

- Pickle

---

## Project Structure

text
Smart_Lender/
│
├── Dataset/
│   └── loan_prediction.csv
│
├── Flask/
│   └── templates/
│       ├── home.html
│       ├── predict.html
│       └── output.html
│
├── app.py
├── app1.py
├── rdf.pkl
├── scale1.pkl
├── requirements.txt
├── README.md
└── .gitignore


---

## Dataset Features

The dataset contains the following input features:
## Dataset Features

The dataset contains the following input features used to predict the loan approval status.

| Feature                       | Data Type                  |                  Description  |
|-------------------------------|----------------------------|-------------------------------|------
| Gender                        | Categorical                       | Gender of the loan applicant (Male/Female). |
| Married                       | Categorical                       | Marital status of the applicant (Yes/No). |
| Dependents                    | Categorical                       | Number of dependents of the applicant (0, 1, 2, 3+). |
| Education                     | Categorical                       | Education qualification of the applicant (Graduate/Not Graduate). |
| Self_Employed                 | Categorical                       | Employment status of the applicant (Yes/No). |
| ApplicantIncome               | Numerical                         | Monthly income of the primary applicant. |
| CoapplicantIncome             | Numerical                         | Monthly income of the co-applicant. |
| LoanAmount                    | Numerical                         | Loan amount requested by the applicant. |
| Loan_Amount_Term              | Numerical                         | Loan repayment duration in months. |
| Credit_History                | Categorical                       | Credit repayment history (1 = Good, 0 = Poor). |
| Property_Area                 | Categorical                       | Location of the property (Urban, Semiurban, Rural). |


### Target Variable

| Value | Description     |
|-------|---------------- |
| 1     | Loan Approved   |
| 0     | Loan Rejected   |

---

## Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Handle Missing Values
   │
   ▼
Encode Categorical Data
   │
   ▼
Feature Scaling
   │
   ▼
Train-Test Split
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Save Model
   │
   ▼
Flask Web Application
```

---

## Machine Learning Algorithms

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier

---

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- F1 Score

---

## Web Application Pages

### Home Page

- Project Overview
- Features
- Technologies Used

### Prediction Page

- Personal Information
- Financial Information
- Credit Information

### Result Page

- Loan Approval Status
- Prediction Result

---

## Installation


### Move to the Project Directory

```bash
cd Smart_Lender
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

### Open in Browser

```text
http://127.0.0.1:5000
```


## Future Enhancements

- User Authentication
- Loan EMI Calculator
- Database Integration
- Admin Dashboard
- Loan History
- PDF Report Generation
- Cloud Deployment
- Explainable AI
- Deep Learning Models

---

## Learning Outcomes

This project demonstrates:

- Data Cleaning
- Feature Engineering
- Feature Scaling
- Machine Learning Classification
- Model Evaluation
- Flask Web Development
- Model Deployment
- Frontend Development

---

## Developer

**Rajarapu Venkateswarlu**

B.Tech – Computer Science and Engineering (Data Science)

NRI Institute of Technology, Guntur

India

---

## License

This project is developed for educational and learning purposes.

Licensed under the MIT License.
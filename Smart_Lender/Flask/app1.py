# import os
# os.system("cls")  # Clears the terminal screen (Windows-only command; use "clear" on Linux/Mac)


# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn
# import imblearn

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     f1_score
# )


# # ==========================================
# # Load Dataset
# # ==========================================
# # Reads the loan prediction CSV into a DataFrame.
# # Note: the backslash in "Dataset\loan_prediction.csv" works on Windows but
# # will break on Linux/Mac. Safer to use a raw string or forward slash:
# # pd.read_csv("Dataset/loan_prediction.csv")
# data = pd.read_csv("Dataset\loan_prediction.csv")
# data.drop("Loan_ID", axis=1, inplace=True)


# # ==========================================
# # Handling Categorical Features
# # ==========================================
# # Converting text/categorical columns into numeric codes so the ML models
# # (which only accept numbers) can use them.

# # Gender: Female -> 1, Male -> 0
# data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})

# # Property Area: ranked/ordinal-style encoding (not strictly ordinal, but
# # treated as numeric here for simplicity)
# data['Property_Area'] = data['Property_Area'].map({
#     'Urban': 2,
#     'Semiurban': 1,
#     'Rural': 0
# })

# # Married: Yes -> 1, No -> 0
# data['Married'] = data['Married'].map({
#     'Yes': 1,
#     'No': 0
# })

# # Education: Graduate -> 1, Not Graduate -> 0
# data['Education'] = data['Education'].map({
#     'Graduate': 1,
#     'Not Graduate': 0
# })

# # Loan Status (this is the TARGET variable we want to predict): Y -> 1, N -> 0
# data['Loan_Status'] = data['Loan_Status'].map({
#     'Y': 1,
#     'N': 0
# })

# # View first 5 rows to sanity-check the encoding worked
# print(data.head())


# # ==========================================
# # Finding Missing Values
# # ==========================================
# # Count how many NaN/null values exist in each column, so we know what
# # needs to be cleaned before modeling.
# print(data.isnull().sum())


# # ==========================================
# # Fill Missing Values
# # ==========================================
# # Strategy: use the mode (most frequent value) for categorical columns,
# # and the median for numeric/continuous columns (median is more robust
# # to outliers than the mean).

# # Fill missing Gender with the most common gender in the dataset
# data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])

# # Fill missing Married status with the most common value
# data['Married'] = data['Married'].fillna(data['Married'].mode()[0])

# # Remove '+' from Dependents (e.g. "3+" -> "3") so it can be converted to int later
# data['Dependents'] = data['Dependents'].str.replace('+', '', regex=False)

# # Fill missing Dependents with the most common value
# data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])

# # Fill missing Self_Employed with the most common value
# data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

# # Fill missing LoanAmount with the median loan amount (numeric column)
# data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())

# # Fill missing Loan_Amount_Term with the most common term length
# data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])

# # Fill missing Credit_History with the most common value (0 or 1)
# data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])


# # ==========================================
# # Check Missing Values Again
# # ==========================================
# # Confirms all NaNs have been handled - should print all 0s now.
# print(data.isnull().sum())


# # ==========================================
# # Dataset Information
# # ==========================================
# # Shows column dtypes, non-null counts, and memory usage.
# print(data.info())


# # ==========================================
# # Type Conversions
# # ==========================================
# # Convert previously-object/float columns into proper integer types now
# # that missing values have been filled and text values encoded.
# data['Gender'] = data['Gender'].astype('int64')
# data['Married'] = data['Married'].astype('int64')
# data['Dependents'] = data['Dependents'].astype('int64')

# # ============================================================
# # ENCODE SELF_EMPLOYED
# # ============================================================
# # (Redundant fillna call here since it was already filled above, but kept
# # as a safety net in case this block runs independently.)
# data['Self_Employed'] = data['Self_Employed'].fillna(
#     data['Self_Employed'].mode()[0]
# )

# # Convert Self_Employed text to numeric: No -> 0, Yes -> 1
# data['Self_Employed'] = data['Self_Employed'].map({
#     'No': 0,
#     'Yes': 1
# })

# data['Self_Employed'] = data['Self_Employed'].astype(int)
# data['LoanAmount'] = data['LoanAmount'].astype('int64')
# data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype('int64')
# data['Credit_History'] = data['Credit_History'].astype('int64')


# # ==========================================
# # Exploratory Data Analysis (EDA) - Distribution Plots
# # ==========================================
# # distplot shows the distribution (histogram + density curve) of a
# # numeric column, useful for spotting skewness or outliers.
# plt.figure(figsize=(12, 5))

# plt.subplot(121)  # 1 row, 2 columns, plot 1: ApplicantIncome distribution
# sns.histplot(
#     data['ApplicantIncome'],
#     kde=True,
#     color='red'
# )
# plt.subplot(122)  # 1 row, 2 columns, plot 2: Credit_History distribution
# sns.histplot(
#     data['Credit_History'],
#     kde=True
# )

# plt.show()

# # ==========================================
# # EDA - Count Plots (Categorical Frequency)
# # ==========================================
# # countplot shows how many records fall into each category.
# plt.figure(figsize=(18, 4))

# plt.subplot(1, 4, 1)  # 1 row, 4 columns, plot 1: Gender counts
# sns.countplot(
#     x='Gender',
#     data=data
# )
# plt.subplot(1, 4, 2)  # 1 row, 4 columns, plot 2: Education counts
# sns.countplot(
#     x='Education',
#     data=data
# )
# plt.show()


# # ==========================================
# # EDA - Comparing Two Columns at Once
# # ==========================================
# # Using 'hue' to break each bar down further by a second category,
# # revealing relationships between two categorical variables.
# plt.figure(figsize=(20, 5))

# plt.subplot(131)  # Married counts, split by Gender
# sns.countplot(x='Married', hue='Gender', data=data)

# plt.subplot(132)  # Self_Employed counts, split by Education
# sns.countplot(x='Self_Employed', hue='Education', data=data)

# plt.subplot(133)  # Property_Area counts, split by Loan_Amount_Term
# sns.countplot(x='Property_Area', hue='Loan_Amount_Term', data=data)

# plt.show()


# # ==========================================
# # EDA - Swarm Plot: Gender & Income vs Loan Status
# # ==========================================
# # Visualizes each applicant as a point, positioned by Gender (x-axis) and
# # ApplicantIncome (y-axis), colored by whether their loan was approved
# # (Loan_Status). Helps spot whether income/gender relate to approval.
# plt.figure(figsize=(8, 6))
# sns.stripplot(
#     x='Gender',
#     y='ApplicantIncome',
#     hue='Loan_Status',
#     data=data,
#     jitter=True,
#     alpha=0.6
# )
# plt.show()

# # -------------------------------------------------------------------------------

# # ============================================================
# # CREATE FEATURES (X) AND TARGET (y)
# # ============================================================

# X = data.drop("Loan_Status", axis=1)
# y = data["Loan_Status"]


# # ============================================================
# # APPLY SMOTE
# # ============================================================

# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=42)

# X_bal, y_bal = smote.fit_resample(X, y)

# print("Before SMOTE")
# print(y.value_counts())

# print("\nAfter SMOTE")
# print(y_bal.value_counts())


# # ============================================================
# # STORE COLUMN NAMES
# # ============================================================

# names = X_bal.columns


# # ============================================================
# # FEATURE SCALING
# # ============================================================

# sc = StandardScaler()

# X_bal = sc.fit_transform(X_bal)

# X_bal = pd.DataFrame(X_bal, columns=names)

# # Save the scaler
# pickle.dump(sc, open("scale1.pkl", "wb"))

# print("Scaler Saved Successfully")

# # ============================================================
# # SPLITTING THE DATASET INTO TRAINING AND TESTING SETS
# # ============================================================

# # Split the balanced dataset into training and testing datasets.
# # 67% of the data is used for training.
# # 33% of the data is used for testing.
# # random_state=42 ensures the same split every time the code runs.

# X_train, X_test, y_train, y_test = train_test_split(
#     X_bal,
#     y_bal,
#     test_size=0.33,
#     random_state=42
# )

# # Display the shape of the training dataset.

# print("Training Feature Shape:")
# print(X_train.shape)

# # Display the shape of the testing dataset.

# print("\nTesting Feature Shape:")
# print(X_test.shape)

# # Display the shape of the training target.

# print("\nTraining Target Shape:")
# print(y_train.shape)

# # Display the shape of the testing target.

# print("\nTesting Target Shape:")
# print(y_test.shape)
# # -------------------------------------------------------------------------------


# # ============================================================
# # DECISION TREE CLASSIFIER
# # ============================================================

# def DecisionTreeModel(X_train, X_test, y_train, y_test):

#     model = DecisionTreeClassifier(random_state=42)

#     model.fit(X_train, y_train)

#     # Training Prediction
#     y_train_pred = model.predict(X_train)

#     print("\n========== Decision Tree ==========")
#     print("Training Accuracy :", accuracy_score(y_train, y_train_pred))

#     # Testing Prediction
#     y_test_pred = model.predict(X_test)

#     print("Testing Accuracy  :", accuracy_score(y_test, y_test_pred))

#     print("\nClassification Report")
#     print(classification_report(y_test, y_test_pred))

#     print("Confusion Matrix")
#     print(confusion_matrix(y_test, y_test_pred))

#     return model


# # Train Decision Tree
# dt_model = DecisionTreeModel(X_train, X_test, y_train, y_test)


# # ============================================================
# # RANDOM FOREST CLASSIFIER
# # ============================================================

# def RandomForestModel(X_train, X_test, y_train, y_test):

#     model = RandomForestClassifier(
#         n_estimators=100,
#         random_state=42
#     )

#     model.fit(X_train, y_train)

#     # Training Prediction
#     y_train_pred = model.predict(X_train)

#     print("\n========== Random Forest ==========")
#     print("Training Accuracy :", accuracy_score(y_train, y_train_pred))

#     # Testing Prediction
#     y_test_pred = model.predict(X_test)

#     print("Testing Accuracy  :", accuracy_score(y_test, y_test_pred))

#     print("\nClassification Report")
#     print(classification_report(y_test, y_test_pred))

#     print("Confusion Matrix")
#     print(confusion_matrix(y_test, y_test_pred))

#     return model


# # Train Random Forest
# rf_model = RandomForestModel(X_train, X_test, y_train, y_test)


# # ============================================================
# # K-NEAREST NEIGHBORS (KNN)
# # ============================================================

# def KNNModel(X_train, X_test, y_train, y_test):

#     model = KNeighborsClassifier()

#     model.fit(X_train, y_train)

#     # Training Prediction
#     y_train_pred = model.predict(X_train)

#     print("\n========== KNN ==========")
#     print("Training Accuracy :", accuracy_score(y_train, y_train_pred))

#     # Testing Prediction
#     y_test_pred = model.predict(X_test)

#     print("Testing Accuracy  :", accuracy_score(y_test, y_test_pred))

#     print("\nClassification Report")
#     print(classification_report(y_test, y_test_pred))

#     print("Confusion Matrix")
#     print(confusion_matrix(y_test, y_test_pred))

#     return model


# # Train KNN
# knn_model = KNNModel(X_train, X_test, y_train, y_test)


# # ============================================================
# # GRADIENT BOOSTING CLASSIFIER
# # ============================================================

# def GradientBoostingModel(X_train, X_test, y_train, y_test):

#     model = GradientBoostingClassifier(random_state=42)

#     model.fit(X_train, y_train)

#     # Training Prediction
#     y_train_pred = model.predict(X_train)

#     print("\n========== Gradient Boosting ==========")
#     print("Training Accuracy :", accuracy_score(y_train, y_train_pred))

#     # Testing Prediction
#     y_test_pred = model.predict(X_test)

#     print("Testing Accuracy  :", accuracy_score(y_test, y_test_pred))

#     print("\nClassification Report")
#     print(classification_report(y_test, y_test_pred))

#     print("Confusion Matrix")
#     print(confusion_matrix(y_test, y_test_pred))

#     return model


# # Train Gradient Boosting
# gb_model = GradientBoostingModel(X_train, X_test, y_train, y_test)


# # ============================================================
# # SAVE THE BEST MODEL
# # ============================================================

# # Save Random Forest model
# pickle.dump(rf_model, open("rdf.pkl", "wb"))

# print("\n Model Saved Successfully.")







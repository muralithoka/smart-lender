"""
Epic 3: Data Pre-processing
Covers: Checking and Handling Missing Values, Balancing the Dataset,
Scaling the Data, Splitting Data into Training and Test Sets
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Story: Checking and Handling Values"""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def balance_dataset(X, y):
    """Story: Balancing the Dataset"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def scale_data(X_train, X_test):
    """Story: Scaling the Data"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Story: Splitting Data into Training and Test Sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    df = load_data("../Epic 1 - Data Collection and Architecture Design/data/loan_data.csv")
    df = handle_missing_values(df)

    # Encode categorical columns, separate X and y as per your dataset schema
    # X = df.drop(columns=["Loan_Status"])
    # y = df["Loan_Status"]

    # X_train, X_test, y_train, y_test = split_data(X, y)
    # X_train, y_train = balance_dataset(X_train, y_train)
    # X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    print("Pre-processing pipeline ready. Fill in dataset-specific column names above.")

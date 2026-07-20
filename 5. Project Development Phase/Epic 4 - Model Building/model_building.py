"""
Epic 4: Model Building
Covers: Decision Tree, Random Forest, KNN, XGBoost models,
and Evaluating Performance and Saving the Model
"""

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def train_decision_tree(X_train, y_train):
    """Story: Decision Tree Model"""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Story: Random Forest Model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    """Story: KNN Model"""
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Story: XGBoost Model"""
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, name: str):
    """Story: Evaluating Performance and Saving the Model"""
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
    print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))
    print(classification_report(y_test, y_pred))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }


def save_model(model, path: str = "model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    print("Import this module's functions after running preprocessing.py to train and compare models.")
    print("Pick the best-performing model and call save_model() to export it for the Flask app.")

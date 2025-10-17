from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n[Logistic Regression]")
    print(classification_report(y_test, preds))
    return model

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n[Random Forest]")
    print(classification_report(y_test, preds))
    return model

def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n[XGBoost]")
    print(classification_report(y_test, preds))
    return model

def train_lightgbm(X_train, y_train, X_test, y_test):
    model = LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n[LightGBM]")
    print(classification_report(y_test, preds))
    return model

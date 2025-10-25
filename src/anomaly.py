import os
import joblib
from tuning import tune_isolation_forest, tune_one_class_svm


def ensure_dirs():
    return "../outputs/models/"

def run_isolation_forest(X_train, X_test, y_test):
    model, best_params, best_auc = tune_isolation_forest(X_train, X_test, y_test)
    models_dir = ensure_dirs()

    model_path = os.path.join(models_dir, "isolation_forest.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    return model, best_params, best_auc


def run_one_class_svm(X_train, X_test, y_test):
    model, best_params, best_auc = tune_one_class_svm(X_train, X_test, y_test)
    models_dir = ensure_dirs()

    model_path = os.path.join(models_dir, "one_class_svm.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    return model, best_params, best_auc

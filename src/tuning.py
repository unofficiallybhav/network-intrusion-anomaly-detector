import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def tune_isolation_forest(X_test, y_test, param_grid=None):
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_samples': ['auto', 0.6, 0.8],
            'contamination': [0.02, 0.05, 0.1],
            'max_features': [0.5, 0.8, 1.0]
        }

    best_auc, best_params = 0, None
    for params in ParameterGrid(param_grid):
        model = IsolationForest(random_state=42, **params)
        preds = model.fit_predict(X_test)
        preds = np.where(preds == -1, 1, 0)
        auc = roc_auc_score(y_test, preds)
        if auc > best_auc:
            best_auc, best_params = auc, params

    print(f"[✔] Best Isolation Forest AUC: {best_auc:.3f}")
    print(f"Best Params: {best_params}")
    return best_params, best_auc


def tune_one_class_svm(X_test, y_test, param_grid=None):
    if param_grid is None:
        param_grid = {
            'kernel': ['rbf', 'sigmoid'],
            'gamma': [0.001, 0.01, 0.1],
            'nu': [0.03, 0.05, 0.1]
        }

    best_auc, best_params = 0, None
    for params in ParameterGrid(param_grid):
        model = OneClassSVM(**params)
        preds = model.fit_predict(X_test)
        preds = np.where(preds == -1, 1, 0)
        auc = roc_auc_score(y_test, preds)
        if auc > best_auc:
            best_auc, best_params = auc, params

    print(f"[✔] Best One-Class SVM AUC: {best_auc:.3f}")
    print(f"Best Params: {best_params}")
    return best_params, best_auc
from preprocessing import load_unsw_data, preprocess_data
from models import train_random_forest, train_xgboost,train_lightgbm,train_logistic_regression
from anomaly import run_isolation_forest, run_one_class_svm
from explainability import explain_with_shap
from selection import apply_pca

def main():
    print("[1] Loading and Preprocessing Data...")
    df = load_unsw_data("../data/")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("[2] Training Supervised Models...")
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
    logistic_regression_model=train_logistic_regression(X_train, y_train, X_test, y_test)

    print("[3] Visualizing PCA...")
    X_train,X_test,_=apply_pca(X_train, y_train,n_components=15)

    print("\n[4] Running Unsupervised Anomaly Detection Models...")
    iso_model, iso_auc = run_isolation_forest(X_train, X_test, y_test)
    ocsvm_model, ocsvm_auc = run_one_class_svm(X_train, X_test, y_test)

    print("\n[5] Model Explainability (SHAP)...")
    explain_with_shap(rf_model, X_test)

    
if __name__ == "__main__":
    main()

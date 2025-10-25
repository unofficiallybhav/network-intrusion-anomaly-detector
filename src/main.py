from preprocessing import load_unsw_data, preprocess_data
from models import train_random_forest, train_xgboost,train_lightgbm,train_logistic_regression
from anomaly import run_isolation_forest, run_one_class_svm
from explainability import explain_with_shap
from selection import apply_pca,unsupervised_feature_selection
from sklearn.utils import resample


def main():
    print("[1] Loading and Preprocessing Data...")
    df = load_unsw_data("../data/")
    X_train, X_test, y_train, y_test = preprocess_data(
        df, 
        model_type='supervised',
        apply_smote=True,
        smote_strategy=0.3
    )

    X_train_unsup,X_test_unsup,y_train_unsup,y_test_unsup=preprocess_data(
        df,
        model_type='unsupervised',
        apply_smote=False
    )

    print("[2] Training Supervised Models...")
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
    logistic_regression_model=train_logistic_regression(X_train, y_train, X_test, y_test)

    print("\n[3] Running Unsupervised Anomaly Detection Models...")
    X_train = unsupervised_feature_selection(X_train_unsup, threshold=0.01)
    X_test = X_test_unsup[X_train.columns]

    X_train,X_test,_=apply_pca(X_train,X_test,n_components=25)

    iso_model , iso_params , iso_auc = run_isolation_forest(X_train, X_test, y_test_unsup)

    X_train_small = resample(X_train, n_samples=10000, random_state=42)
    ocsvm_model , ocsvm_params , ocsvm_auc = run_one_class_svm(X_train_small, X_test, y_test_unsup)

    # print("\n[4] Model Explainability (SHAP)...")
    # explain_with_shap(rf_model, X_test)

    
if __name__ == "__main__":
    main()

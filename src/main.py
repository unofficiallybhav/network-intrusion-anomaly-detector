from preprocessing import load_unsw_data, preprocess_data
from feature import visualize_pca
from models import train_random_forest, train_xgboost
from anomaly import isolation_forest, one_class_svm
from explainability import explain_with_shap
from utils import save_model

def main():
    print("[1] Loading and Preprocessing Data...")
    df = load_unsw_data("../data/")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("[2] Visualizing PCA...")
    visualize_pca(X_train, y_train)

    print("[3] Training Supervised Models...")
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    print("[4] Running Unsupervised Anomaly Detection...")
    isolation_forest(X_test, y_test)
    one_class_svm(X_test, y_test)

    print("[5] Model Explainability (SHAP)...")
    explain_with_shap(rf_model, X_test)

if __name__ == "__main__":
    main()

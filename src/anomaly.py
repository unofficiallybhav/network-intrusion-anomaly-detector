import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tuning import tune_isolation_forest, tune_one_class_svm
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import joblib

def run_isolation_forest(X_train, X_test, y_test):

    best_params, _ = tune_isolation_forest(X_test, y_test)

    iso = IsolationForest(random_state=42, **best_params)
    preds = iso.fit_predict(X_test)
    preds = np.where(preds == -1, 1, 0)
    auc = roc_auc_score(y_test, preds)

    print(f"Final Isolation Forest ROC-AUC: {auc:.3f}")
    joblib.dump(iso, r"C:\Users\hp\OneDrive\Desktop\Python\Machine Learning\project\outputs\models\isolation_forest.pkl" )
    print(f"[💾] Saved model")
    return iso, auc


def run_one_class_svm(X_train, X_test, y_test):

    best_params, _ = tune_one_class_svm(X_test, y_test)

    ocsvm = OneClassSVM(**best_params)
    preds = ocsvm.fit_predict(X_test)
    preds = np.where(preds == -1, 1, 0)
    auc = roc_auc_score(y_test, preds)

    print(f"Final One-Class SVM ROC-AUC: {auc:.3f}")
    joblib.dump(ocsvm, r"C:\Users\hp\OneDrive\Desktop\Python\Machine Learning\project\outputs\models\one_class_svm.pkl" )
    print(f"[💾] Saved model")
    return ocsvm, auc

def dbscan_cluster(X_pca):
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    clusters = dbscan.fit_predict(X_pca)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='tab10', s=8)
    plt.title("DBSCAN Clustering of Traffic")
    plt.show()
    return dbscan

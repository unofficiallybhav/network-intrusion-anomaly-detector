import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def isolation_forest(X_test, y_test):
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(X_test)
    preds = np.where(preds == -1, 1, 0)
    auc = roc_auc_score(y_test, preds)
    print(f"Isolation Forest ROC-AUC: {auc:.3f}")
    return iso

def one_class_svm(X_test, y_test):
    ocsvm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05)
    preds = ocsvm.fit_predict(X_test)
    preds = np.where(preds == -1, 1, 0)
    auc = roc_auc_score(y_test, preds)
    print(f"One-Class SVM ROC-AUC: {auc:.3f}")
    return ocsvm

def dbscan_cluster(X_pca):
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    clusters = dbscan.fit_predict(X_pca)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='tab10', s=8)
    plt.title("DBSCAN Clustering of Traffic")
    plt.show()
    return dbscan

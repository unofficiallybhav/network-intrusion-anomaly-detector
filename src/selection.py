import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def remove_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    print(f"[✔] Dropped {len(to_drop)} correlated features.")
    return X_reduced, to_drop


def select_top_features(X_train, y_train, top_n=25):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()
    print(f"[✔] Selected top {top_n} features using RandomForest importance.")
    return top_features, importances


def apply_pca(X_train, X_test, n_components=15):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"[✔] Applied PCA → {n_components} components explain {sum(pca.explained_variance_ratio_):.2%} variance.")
    return X_train_pca, X_test_pca, pca
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

def remove_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    return X_reduced, to_drop


def unsupervised_feature_selection(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    selected = X.columns[selector.get_support()]
    print(f"Retained {len(selected)} high-variance features out of {X.shape[1]}")
    return pd.DataFrame(X_reduced, columns=selected)


def apply_pca(X_train, X_test, n_components=15):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"Applied PCA → {n_components} components explain {sum(pca.explained_variance_ratio_):.2%} variance.")
    return X_train_pca, X_test_pca, pca
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def correlation_analysis(df, threshold=0.9):
    corr = df.corr().abs()
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(df[to_drop], axis=1)
    return df, to_drop

def visualize_pca(X, y, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=10)
    plt.title("PCA Visualization of Network Traffic")
    plt.show()
    return pca

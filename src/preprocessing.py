import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from selection import remove_correlated_features


def load_unsw_data(data_path="../data/", columns_file=None):
    if columns_file is None:
        columns_file = os.path.join(data_path, "NUSW-NB15_features.csv")

    # Handle multiple possible encodings
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            cols_df = pd.read_csv(columns_file, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    # Extract column names
    if 'Name' in cols_df.columns:
        column_names = cols_df['Name'].tolist()
    elif 'Feature' in cols_df.columns:
        column_names = cols_df['Feature'].tolist()
    else:
        column_names = cols_df.iloc[:, 0].tolist()

    # Merge all 4 CSV parts
    csv_files = [os.path.join(data_path, f"UNSW-NB15_{i}.csv") for i in range(1, 5)]
    df = pd.concat(
        (pd.read_csv(f, header=None, names=column_names, low_memory=False) for f in csv_files),
        ignore_index=True
    )

    print(f"[✔] Loaded UNSW-NB15 dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df, test_size=0.3, random_state=42, drop_corr=True, corr_threshold=0.9):
    # Drop duplicates and fill NaNs
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Drop attack_cat to prevent label leakage
    if "attack_cat" in df.columns:
        df.drop(columns=["attack_cat"], inplace=True)
        print("[✔] Dropped 'attack_cat' column (prevented leakage).")

    # Detect target column
    target_col = None
    for col in ["label", "Label"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"[❌] Target column ('label' or 'Label') not found in dataset.")

    # Encode categorical variables (excluding target)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col != target_col:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Encode target if needed
    if df[target_col].dtype == "object":
        df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))

    # Separate features and labels
    X = df.drop(columns=[target_col])
    y = df[target_col]

    #Detect and drop highly correlated features
    _,correlated = remove_correlated_features(df, target_col=target_col, threshold=corr_threshold)
    if correlated:
        X.drop(columns=correlated, inplace=True)
        print(f"[✔] Dropped {len(correlated)} highly correlated feature(s): {correlated}")

    # Split before scaling (avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale using only training data statistics
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(f"Data split complete → Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Label distribution (train):\n{y_train.value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test

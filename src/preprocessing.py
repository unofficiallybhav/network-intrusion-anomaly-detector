import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_unsw_data(data_path="../data/", columns_file="../data/NUSW-NB15_features.csv"):
    # Read column names from the features file with encoding handling
    try:
        cols_df = pd.read_csv(columns_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            cols_df = pd.read_csv(columns_file, encoding='latin-1')
        except UnicodeDecodeError:
            cols_df = pd.read_csv(columns_file, encoding='cp1252')
    
    # Extract column names - adjust based on your file structure
    # Common formats: single column with names, or 'Name'/'Feature' column
    if 'Name' in cols_df.columns:
        column_names = cols_df['Name'].tolist()
    elif 'Feature' in cols_df.columns:
        column_names = cols_df['Feature'].tolist()
    else:
        # If it's just a single column, use the first column
        column_names = cols_df.iloc[:, 0].tolist()
    
    files = [f"{data_path}UNSW-NB15_{i}.csv" for i in range(1,5)]
    # Fix: No header in CSV files, specify column names from file
    df = pd.concat((pd.read_csv(f, header=None, names=column_names, low_memory=False) 
                    for f in files), ignore_index=True)
    return df

def preprocess_data(df):
    # Drop duplicates, handle NaNs
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Find the target column (could be 'label', 'Label', 'attack_cat', etc.)
    target_col = None
    for col in ['label', 'Label', 'attack_cat']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Target column not found. Available columns: {df.columns.tolist()}")

    # Encode categorical columns (except target)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != target_col:
            # Fix: Convert mixed-type columns to strings first
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Encode target column if it's categorical
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype(str)
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col])

    # Separate features and target
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    # Scale numeric features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
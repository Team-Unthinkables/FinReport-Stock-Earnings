# In preprocessing.py

from sklearn.preprocessing import StandardScaler, LabelEncoder

def rename_technical_columns(df):
    """
    Renames columns that look like tuple strings into a simpler format.
    For example, convert "('technical_indicators_overbought_oversold', 'RSI')" to 
    "technical_indicators_overbought_oversold_RSI".
    """
    new_columns = {}
    for col in df.columns:
        # Check if the column is a string that looks like a tuple.
        if isinstance(col, str) and col.startswith("(") and col.endswith(")"):
            # Remove parentheses and extra quotes, then replace comma with underscore.
            new_col = col.strip("()").replace("'", "").replace(", ", "_")
            new_columns[col] = new_col
    df = df.rename(columns=new_columns)
    # Debug: Print renamed columns
    print("Renamed columns:", df.columns.tolist())
    return df

def encode_ts_code(df):
    # Create an explicit copy to avoid SettingWithCopyWarning
    df = df.copy()
    le = LabelEncoder()
    df['ts_code_encoded'] = le.fit_transform(df['ts_code'])
    return df, le

def select_features(df):
    # First, rename technical indicator columns.
    df = rename_technical_columns(df)
    
    # Exclude columns such as 'date', 'announcement', and any other non-numeric ones.
    exclude_cols = ['date', 'announcement']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    # Now only take numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    features = df[numeric_cols].values
    targets = df['label'].values  # assuming 'label' is numeric
    
    # Ensure features and labels arrays are created from the same rows.
    assert len(features) == len(targets), "Features and labels must have the same length"
    
    return features, targets

def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler

if __name__ == "__main__":
    import pandas as pd
    test_df = pd.DataFrame({
        "('technical_indicators_overbought_oversold', 'RSI')": [50],
        "date": ["2021-01-01"],
        "announcement": ["Sample announcement"],
        "label": [1]
    })
    test_df = rename_technical_columns(test_df)
    print("Test DataFrame after renaming:")
    print(test_df.head())

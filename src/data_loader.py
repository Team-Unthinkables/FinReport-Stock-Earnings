import pandas as pd
import ast

def try_literal_eval(val):
    if isinstance(val, str) and (val.startswith('[') or val.startswith('{')):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    return val

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Apply conditional parsing to the 'announcement' column:
    df['announcement'] = df['announcement'].apply(try_literal_eval)
    return df

def split_data(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df

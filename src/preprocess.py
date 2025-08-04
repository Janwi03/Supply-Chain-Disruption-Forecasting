import pandas as pd
from sklearn.utils import resample

def load_data(path):
    """Load CSV and drop empty trailing row if exists."""
    df = pd.read_csv(path)
    df = df[~df.isnull().all(axis=1)]
    df.drop(df.tail(1).index, inplace=True)
    return df

def encode_binaries(df):
    """Map Yes/No binary columns to 1/0."""
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if 'Yes' in df[col].unique() or 'No' in df[col].unique():
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    return df

def sample_data(df, target_col='went_on_backorder', majority_count=26350):
    """Sample to achieve approx 70:30 class balance."""
    class_0 = df[df[target_col] == 0]
    class_1 = df[df[target_col] == 1]

    class_0_sample = resample(class_0,
                              replace=False,
                              n_samples=majority_count,
                              random_state=42)

    df_sampled = pd.concat([class_0_sample, class_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_sampled

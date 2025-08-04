import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def check_binary_feature_balance(df, binary_cols):
    for col in binary_cols:
        counts = df[col].value_counts(normalize=True)
        print(f"{col}:\n{counts}\n")

def knn_impute(df, n_neighbors=5):
    X_numeric = df.select_dtypes(include=['int64', 'float64'])
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_numeric), columns=X_numeric.columns)
    return X_imputed, imputer

def calculate_vif(X):
    X_with_const = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                       for i in range(X_with_const.shape[1])]
    return vif_data[vif_data["Feature"] != "const"].sort_values(by="VIF", ascending=False)

def plot_corr_heatmap(X, title="Spearman Correlation Heatmap", threshold=None):
    corr = X.corr(method='spearman')
    if threshold:
        corr = corr[corr.abs() > threshold]
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def drop_high_corr_features(df, features_to_drop):
    return df.drop(columns=features_to_drop)

def split_and_scale(X, y):
    selected_features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler, selected_features


def full_model_preprocessing_pipeline(df_sampled):
    binary_cols = ['deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'potential_issue']
    check_binary_feature_balance(df_sampled, binary_cols)

    X = df_sampled.drop('went_on_backorder', axis=1)
    y = df_sampled['went_on_backorder']
    X_imputed, _ = knn_impute(X)

    print("\n VIF Before Dropping:")
    print(calculate_vif(X_imputed))

    plot_corr_heatmap(X_imputed[[
        'forecast_3_month', 'forecast_6_month', 'forecast_9_month',
        'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
        'perf_6_month_avg', 'perf_12_month_avg'
    ]])

    to_drop = [
        'rev_stop','oe_constraint','potential_issue','stop_auto_buy',
        'forecast_3_month','forecast_9_month',
        'sales_1_month','sales_3_month','sales_9_month',
        'perf_6_month_avg'
    ]
    df_reduced = df_sampled.drop(columns=to_drop)
    X_new = df_reduced.drop('went_on_backorder', axis=1)
    y_new = df_reduced['went_on_backorder']

    # Return updated split and preprocessing objects
    return split_and_scale(X_new, y_new)


from src.preprocess import load_data, encode_binaries, sample_data

def prepare_data(csv_path="data/raw/Training_BOP.csv"):
    raw_df = load_data(csv_path)
    encoded_df = encode_binaries(raw_df)
    df_sampled = sample_data(encoded_df)

    return full_model_preprocessing_pipeline(df_sampled)
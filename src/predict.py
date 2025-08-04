import pandas as pd
import joblib
import os

def load_model_artifacts(model_path, imputer_path, scaler_path, feature_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(imputer_path):
        raise FileNotFoundError(f"Imputer file not found: {imputer_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature list file not found: {feature_path}")

    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(feature_path)
    return model, imputer, scaler, selected_features

def preprocess_input_data(input_df, imputer, scaler, selected_features):
    # Encode Yes/No binary columns like in training
    binary_cols = ['deck_risk', 'oe_constraint', 'ppap_risk',
                   'stop_auto_buy', 'rev_stop', 'potential_issue']
    for col in binary_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    # Drop non-numeric, unused columns like 'sku' if not in selected_features
    for col in input_df.columns:
        if col not in selected_features:
            input_df.drop(col, axis=1, inplace=True)

    # Reorder columns to match training order
    missing_cols = set(selected_features) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input data: {list(missing_cols)}")

    input_df = input_df[selected_features]

    # Impute and scale
    imputed = imputer.transform(input_df)
    scaled = scaler.transform(imputed)

    return scaled


def predict_from_csv(model_path, imputer_path, scaler_path, feature_path, csv_path, sample_size=None):
    print("\nReading input data")
    df = pd.read_csv(csv_path)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    print("Loading model and preprocessing artifacts")
    model, imputer, scaler, selected_features = load_model_artifacts(
        model_path, imputer_path, scaler_path, feature_path
    )

    print("Preprocessing input data")
    X_processed = preprocess_input_data(df, imputer, scaler, selected_features)

    print("Making predictions")
    preds = model.predict(X_processed)
    df['Prediction'] = preds
    return df

# Run prediction if file executed directly
if __name__ == "__main__":
    MODEL_PATH = "models/rf_best_model.joblib"
    IMPUTER_PATH = "models/imputer.joblib"
    SCALER_PATH = "models/scaler.joblib"
    FEATURE_PATH = "models/selected_features.joblib"
    CSV_PATH = r"data/raw/Training_BOP.csv"

    output_df = predict_from_csv(MODEL_PATH, IMPUTER_PATH, SCALER_PATH, FEATURE_PATH, CSV_PATH, sample_size=50000)

    os.makedirs("data", exist_ok=True)
    output_df.to_csv("data/predictions.csv", index=False)
    print("\n Predictions saved to data/predictions.csv")
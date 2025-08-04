import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
import optuna
import joblib
from src.threshold_tuning import threshold_tuning


def train_models(X_train, X_test, y_train, y_test):
    models = {
        'AdaBoost': AdaBoostClassifier(),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression()
    }

    all_preds = {}

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        all_preds[name] = preds

        print(f"\n Predictions ({name}): {preds[:10]}")
        print(f"\n Classification Report ({name}):")
        print(classification_report(y_test, preds, zero_division=0))

        cm = confusion_matrix(y_test, preds)
        print(f"\n Confusion Matrix ({name}):\n{pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1'])}")
    
    return models


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        avg_precision = average_precision_score(y_test, y_probs)
        plt.plot(recall, precision, label=f"{name} (AP = {avg_precision:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for All Models")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def tune_random_forest(X_train, y_train, X_test, y_test):
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
        }
        model = RandomForestClassifier(**params, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1).mean()

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=20)
    best_params = study_rf.best_params

    rf_best = RandomForestClassifier(**best_params, random_state=42)
    rf_best.fit(X_train, y_train)
    evaluate_model(rf_best, X_train, y_train, X_test, y_test, name="Random Forest (Optuna)")

    joblib.dump(rf_best, "models/rf_best_model.joblib")
    return rf_best


def tune_xgboost(X_train, y_train, X_test, y_test):
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
        }
        model = XGBClassifier(**params, eval_metric='logloss', random_state=42)
        return cross_val_score(model, X_train, y_train, cv=3, scoring='f1').mean()

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=30)
    best_params = study_xgb.best_params

    xgb_best = XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
    xgb_best.fit(X_train, y_train)
    evaluate_model(xgb_best, X_train, y_train, X_test, y_test, name="XGBoost (Optuna)")
    return xgb_best


def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    print(f"\n--- {name} ---")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Train Classification Report:\n", classification_report(y_train, y_train_pred))
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


if __name__ == "__main__":
    from src.preprocess_modeling import prepare_data

    # Get processed data + preprocessing objects
    X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler, selected_features = prepare_data()

    # Train and save best model
    rf_best = tune_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)

    # Save preprocessing objects and features used during training
    joblib.dump(imputer, "models/imputer.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(selected_features, "models/selected_features.joblib")
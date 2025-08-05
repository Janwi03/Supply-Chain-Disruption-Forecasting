# 📦 Supply Chain Disruption Forecasting

Predicting supply chain disruptions before they happen is critical for ensuring continuity and customer satisfaction in manufacturing and logistics. This project aims to forecast whether a product will go on backorder based on historical supply chain features. It combines end-to-end data preprocessing, feature selection, model training, hyperparameter tuning, and deployment through a user-friendly Streamlit interface.

---

## 🔧 Key Features

* **Exploratory Data Analysis & VIF-based Feature Engineering**
* **KNN Imputation** for handling missing values
* **StandardScaler** for normalization
* **Machine Learning Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, AdaBoost
* **Hyperparameter Tuning** using **Optuna**
* **Model Evaluation** via ROC-AUC, Precision-Recall Curves, and F1-Score
* **Streamlit Dashboard** for real-time prediction and interaction
* **Modular and Scalable Codebase** suitable for production pipelines

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Janwi03/Supply-Chain-Disruption-Forecasting.git
cd Supply-Chain-Disruption-Forecasting
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Supply_Chain_Prediction/
├── data/
│   ├── raw/                        <- Raw CSV files
│   └── predictions.csv            <- Output predictions
├── models/                        <- Saved ML artifacts
│   ├── rf_best_model.joblib
│   ├── imputer.joblib
│   ├── scaler.joblib
│   └── selected_features.joblib
├── notebooks/                     <- Jupyter notebooks for EDA & trials
├── src/                           <- Core source scripts
│   ├── preprocess.py
│   ├── preprocess_modeling.py
│   ├── train.py
│   ├── predict.py
│   └── threshold_tuning.py
├── app.py                         <- Streamlit app
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance

* **Final Model**: Random Forest Classifier
* **Optimization**: Optuna with 20+ trials
* **Metrics**:

  * F1-Score: *insert score here*
  * AUC-ROC: *insert score here*
  * Average Precision: *insert score here*

---

## 🌍 Streamlit App Preview

The app allows users to:

* Input key features affecting supply chain disruptions
* Get real-time predictions with probability
* View predictions as a pie chart for visual clarity

The layout is designed to be clean, intuitive, and executive-friendly.

---

## 🛠️ Tools & Technologies

* **Languages**: Python
* **Libraries**: scikit-learn, xgboost, optuna, pandas, matplotlib, seaborn
* **Visualization**: Matplotlib, Seaborn, Streamlit
* **Deployment**: Streamlit (local, can be pushed to Streamlit Cloud)

---

## 🙋 About the Author

**Janwi Bhattar**
Aspiring Data Scientist | Machine Learning Enthusiast | Passionate about building real-world solutions with data
GitHub: [@Janwi03](https://github.com/Janwi03)

---

## 💌 Feedback & Contributions

Have feedback, suggestions, or want to contribute? Feel free to open an issue or submit a PR. Collaboration is always welcome!

---

> “In supply chains, a small disruption can trigger a ripple effect. Predicting that disruption can prevent the wave.”

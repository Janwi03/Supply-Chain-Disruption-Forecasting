import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# Load artifacts
model = joblib.load("models/rf_best_model.joblib")
imputer = joblib.load("models/imputer.joblib")
scaler = joblib.load("models/scaler.joblib")
selected_features = joblib.load("models/selected_features.joblib")

# App layout
st.set_page_config(page_title="Supply Chain Risk Prediction", layout="wide")
st.sidebar.title("🔍 Supply Chain Input Features")

# Sidebar user inputs
input_data = {}
for feature in selected_features:
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input
imputed = imputer.transform(input_df)
scaled = scaler.transform(imputed)

# Prediction
proba = model.predict_proba(scaled)[0]
pred = model.predict(scaled)[0]

# Main Panel Output
st.title("📦 Supply Chain Disruption Prediction")
st.write("## Prediction Results")

col1, col2 = st.columns([2, 3])

with col1:
    st.metric(label="Prediction", value="Yes" if pred == 1 else "No")
    st.metric(label="Disruption Probability", value=f"{proba[1]:.2f}")

with col2:
    st.write("### Probability Distribution")

    fig = go.Figure(data=[go.Pie(
        labels=["No", "Yes"],
        values=proba,
        hole=0.5,
        marker=dict(colors=["#FF9999", "#66B3FF"]),
        textinfo='label+percent'
    )])

    fig.update_layout(
        title_text='📊 Prediction Confidence',
        width=400,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=False)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
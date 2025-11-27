import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Page setup
st.set_page_config(page_title="Bird Observatory – XAI App", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("F&E_full_dataset.xlsx")

    remove_cols = ["Unnamed: 0", "hitID", "runID", "batchID", "ts",
                   "tsCorrected", "DATE", "TIME", "port", "antBearing"]
    df.drop(columns=[c for c in remove_cols if c in df.columns],
            inplace=True, errors="ignore")

    df["valid"] = (df["motusFilter"] == 1).astype(int)

    features = [
        "sig", "sigsd", "snr", "runLen",
        "avg_sig_per_tag", "avg_snr_per_tag", "detections_per_tag"
    ]

    X = df[features].fillna(df[features].median())
    y = df["valid"]

    return X, y, features, df

X, y, feature_cols, raw_data = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Final tuned Random Forest
rf = RandomForestClassifier(
    n_estimators=120, max_depth=12, random_state=42
)
rf.fit(X_train, y_train)

# App Title
st.title("📡 Bird Observatory – ML Model With XAI")
st.write("This app explains how the ML model predicts valid detections using Explainable AI (XAI).")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Prediction",
    "📊 Feature Importance",
    "🧠 SHAP (Simple)",
    "📘 Model Performance",
    "📄 Data Preview"
])

# -----------------------------------------------------------
# Tab 1 – Prediction with Confidence
# -----------------------------------------------------------
with tab1:
    st.subheader("Make a Prediction")

    inputs = {}
    for col in feature_cols:
        val = st.number_input(
            f"{col}", 
            float(X[col].min()), 
            float(X[col].max()), 
            float(X[col].median())
        )
        inputs[col] = val

    input_df = pd.DataFrame([inputs])

    if st.button("Predict"):
        prob = rf.predict_proba(input_df)[0][1]
        pred = rf.predict(input_df)[0]

        st.write(f"### Prediction: **{'Valid' if pred == 1 else 'Invalid'}**")
        st.write(f"### Confidence: **{prob:.3f}**")

        low = max(prob - 0.10, 0)
        high = min(prob + 0.10, 1)

        st.write("### Confidence Range (±10%)")
        st.progress(prob)
        st.write(f"Lower: **{low:.2f}** | Upper: **{high:.2f}**")

        st.write("This range gives the client an idea of uncertainty in the prediction.")

# -----------------------------------------------------------
# Tab 2 – Feature Importance
# -----------------------------------------------------------
with tab2:
    st.subheader("Feature Importance")

    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=rf.feature_importances_, y=feature_cols, ax=ax)
    plt.title("Feature Importance")
    st.pyplot(fig)

    st.write("""
    **Explanation:**  
    This chart shows which features affect the model the most.  
    Higher bars mean stronger impact on predicting whether a detection is valid.
    """)

# -----------------------------------------------------------
# Tab 3 – SHAP Simple Bar Plot
# -----------------------------------------------------------
with tab3:
    st.subheader("SHAP Value Explanation (Simple Version)")

    st.write("SHAP explains how each feature contributes to predictions.")

    # Small sample for speed
    X_small = X_test.sample(20, random_state=42).values

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_small)

    # SHAP bar plot (works 100% of the time)
    fig, ax = plt.subplots(figsize=(7,5))
    shap.plots.bar(shap_values[1], max_display=7)
    st.pyplot(fig)

    st.write("""
    **SHAP Explanation:**  
    This bar chart shows the average impact of each feature.  
    Positive values push predictions toward **Valid**,  
    negative values push toward **Invalid**.
    """)

# -----------------------------------------------------------
# Tab 4 – Model Performance
# -----------------------------------------------------------
with tab4:
    st.subheader("Model Performance on Test Data")

    preds = rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("Precision", f"{prec:.3f}")
    st.metric("Recall", f"{rec:.3f}")
    st.metric("F1 Score", f"{f1:.3f}")

    st.write("These metrics help validate how well the model performs for the client.")

# -----------------------------------------------------------
# Tab 5 – Show Raw Data
# -----------------------------------------------------------
with tab5:
    st.subheader("Dataset Preview")
    st.dataframe(raw_data.head(50))
    st.write("This helps clients verify where the model's inputs come from.")

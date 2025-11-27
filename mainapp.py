import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Bird Observatory XAI App", layout="wide")

# -----------------------------
# Load & Prepare Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("F&E_full_dataset.xlsx")

    drop_cols = ["Unnamed: 0", "hitID", "runID", "batchID", "ts",
                 "tsCorrected", "DATE", "TIME", "port", "antBearing"]

    df.drop(columns=[c for c in drop_cols if c in df.columns],
            inplace=True, errors="ignore")

    df["valid"] = (df["motusFilter"] == 1).astype(int)

    feature_cols = [
        "sig", "sigsd", "snr", "runLen",
        "avg_sig_per_tag", "avg_snr_per_tag", "detections_per_tag"
    ]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["valid"]
    return X, y, feature_cols, df

X, y, feature_cols, df_raw = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -----------------------------
# Train Final Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=120, max_depth=12, random_state=42
)
rf.fit(X_train, y_train)

# -----------------------------
# App Title
# -----------------------------
st.title("📡 Bird Observatory – ML Insights with Explainable AI (XAI)")
st.write("This app helps explain how the Random Forest model makes predictions.")

# ======================================================
# TABS FOR MODEL OUTPUTS + XAI
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Make Prediction",
    "📊 Feature Importance",
    "🧠 SHAP Values",
    "📘 Model Performance"
])

# ======================================================
# TAB 1 — PREDICTION + CONFIDENCE
# ======================================================
with tab1:
    st.subheader("Enter feature values to predict validity")

    inputs = {}
    for col in feature_cols:
        val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].median()))
        inputs[col] = val

    input_df = pd.DataFrame([inputs])

    if st.button("Predict"):
        prob = rf.predict_proba(input_df)[0][1]
        pred = rf.predict(input_df)[0]

        st.write(f"### Prediction: **{'Valid' if pred == 1 else 'Invalid'}**")
        st.write(f"### Confidence: **{prob:.3f}**")

        # Simple “prediction interval”
        low_bound = max(prob - 0.10, 0)
        high_bound = min(prob + 0.10, 1)

        st.write("#### Confidence Range (±10%)")
        st.progress(prob)
        st.write(f"Lower: {low_bound:.2f}   |   Upper: {high_bound:.2f}")


# ======================================================
# TAB 2 — FEATURE IMPORTANCE
# ======================================================
with tab2:
    st.subheader("Feature Importance")

    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=rf.feature_importances_, y=feature_cols, ax=ax)
    plt.title("Random Forest Feature Importance")
    st.pyplot(fig)

    st.write("""
    **How this helps the client:**  
    This chart explains *which factors influence predictions the most.*  
    Higher bars = features that strongly impact whether a detection is valid.
    """)


# ======================================================
# TAB 3 — SIMPLE SHAP EXPLANATIONS
# ======================================================
with tab3:
    st.subheader("SHAP Value Explanation")

    st.write("SHAP helps show how each feature pushes the prediction up or down.")

    # Small sample for speed
    X_small = X_test.sample(20, random_state=42).values

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_small)

    fig, ax = plt.subplots(figsize=(7,5))
    shap.summary_plot(shap_values[1], X_small, feature_names=feature_cols, show=False)
    st.pyplot(fig)

    st.write("""
    **How this helps the client:**  
    SHAP values show *why* the model predicted "valid" or "invalid".  
    Red = pushes prediction toward valid  
    Blue = pushes prediction toward invalid
    """)


# ======================================================
# TAB 4 — MODEL PERFORMANCE
# ======================================================
with tab4:
    st.subheader("Model Performance")

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    st.write(f"Accuracy: **{acc:.3f}**")
    st.write(f"Precision: **{prec:.3f}**")
    st.write(f"Recall: **{rec:.3f}**")
    st.write(f"F1 Score: **{f1:.3f}**")

    st.write("""
    These metrics help verify that the model performs well  
    and gives confidence in using predictions for decision-making.
    """)

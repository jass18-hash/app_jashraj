import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

st.set_page_config(page_title="Bird Observatory – XAI App", layout="wide")

# -------------------------
# Load data
# -------------------------
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Final tuned RF
rf = RandomForestClassifier(
    n_estimators=120, max_depth=12, random_state=42
)
rf.fit(X_train, y_train)

# Page title
st.title("📡 Bird Observatory – Machine Learning with Explainability")
st.write("This app shows predictions and explains model decision factors using simple XAI.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📊 Feature Importance + Sensitivity",
    "📘 Model Performance",
    "📄 Dataset Preview"
])


# ============================================================
# TAB 1 — Prediction with Confidence
# ============================================================
with tab1:
    st.subheader("Enter feature values to get prediction")

    inputs = {}
    for col in feature_cols:
        inputs[col] = st.number_input(
            f"{col}", 
            float(X[col].min()), 
            float(X[col].max()), 
            float(X[col].median())
        )

    df_input = pd.DataFrame([inputs])

    if st.button("Predict"):
        prob = rf.predict_proba(df_input)[0][1]
        pred = rf.predict(df_input)[0]

        st.write(f"### Prediction: **{'Valid' if pred==1 else 'Invalid'}**")
        st.write(f"### Confidence: **{prob:.3f}**")

        low = max(prob - 0.10, 0)
        high = min(prob + 0.10, 1)

        st.write("### Confidence Range (±10%)")
        st.progress(prob)
        st.write(f"Lower: **{low:.2f}** | Upper: **{high:.2f}**")

        st.info("This confidence range shows prediction uncertainty.")


# ============================================================
# TAB 2 — Feature Importance + Interactive Sensitivity
# ============================================================
with tab2:
    st.subheader("Feature Importance (What Drives the Model)")

    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=rf.feature_importances_, y=feature_cols)
    plt.title("Random Forest Feature Importance")
    st.pyplot(fig)

    st.write("""
    Higher bars mean the model relies more on that feature  
    to decide whether a detection is valid or invalid.
    """)

    st.markdown("---")
    st.subheader("Try Changing One Feature (Sensitivity Test)")

    feature_to_change = st.selectbox("Select a feature:", feature_cols)

    test_row = X_test.sample(1, random_state=42).copy()

    original_value = test_row.iloc[0][feature_to_change]

    new_value = st.slider(
        f"Change value for {feature_to_change}",
        float(X[feature_to_change].min()),
        float(X[feature_to_change].max()),
        float(original_value)
    )

    test_row[feature_to_change] = new_value

    new_prob = rf.predict_proba(test_row)[0][1]

    st.write(f"### New Predicted Probability: **{new_prob:.3f}**")

    st.write("""
    This helps clients understand:  
    *“If we change this value, how does the prediction move?”*
    """)

# ============================================================
# TAB 3 — Model Performance
# ============================================================
with tab3:
    st.subheader("Model Performance Summary")

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("Precision", f"{prec:.3f}")
    st.metric("Recall", f"{rec:.3f}")
    st.metric("F1 Score", f"{f1:.3f}")

    st.write("These scores show how well the model performed on test data.")

# ============================================================
# TAB 4 — Dataset Preview
# ============================================================
with tab4:
    st.subheader("Dataset Sample")
    st.dataframe(raw_data.head(50))

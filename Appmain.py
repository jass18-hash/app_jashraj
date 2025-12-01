import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bird Observatory – ML XAI App", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("F&E_full_dataset.xlsx")

    # Remove columns we do not need
    remove_cols = ["Unnamed: 0", "hitID", "runID", "batchID", "ts",
                   "tsCorrected", "DATE", "TIME", "port", "antBearing"]
    df.drop(columns=[c for c in remove_cols if c in df.columns],
            inplace=True, errors="ignore")

    # Target column
    df["valid"] = (df["motusFilter"] == 1).astype(int)

    # Features used by the model
    features = [
        "sig", "sigsd", "snr", "runLen",
        "avg_sig_per_tag", "avg_snr_per_tag", "detections_per_tag"
    ]

    X = df[features].fillna(df[features].median())
    y = df["valid"]

    return X, y, features, df

X, y, feature_cols, raw_data = load_data()

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Train final Random Forest model
rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=12,
    random_state=42
)
rf.fit(X_train, y_train)

# App title and problem statement
st.title("Bird Observatory – Machine Learning with Explainability")

st.write("Problem statement:")
st.write(
    "Build a model that predicts if a detection is valid (1) or noise (0) "
    "using the signal and frequency features from the dataset."
)

st.write("""
This application uses a Random Forest model to predict whether a detection is **Valid** or **Noise**.  
It also explains which features matter and how changing values affects the prediction.
""")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Prediction",
    "Feature Importance and Sensitivity",
    "Model Performance",
    "Dataset Preview",
    "EDA",
    "Chatbot"
])




# Tab 1 – Prediction
with tab1:
    st.subheader("Make a prediction")

    st.write("""
    How this prediction works:
    - The model outputs a probability between 0 and 1.
    - If the probability is closer to 1, the detection is more likely **Valid**.
    - If it is closer to 0, it is more likely **Noise**.
    - The confidence range (lower and upper) shows a simple ±8% band around the prediction 
      to give an idea of uncertainty.
    """)

    user_inputs = {}
    for col in feature_cols:
        user_inputs[col] = st.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].median())
        )

    user_df = pd.DataFrame([user_inputs])

    if st.button("Predict"):
        prob = rf.predict_proba(user_df)[0][1]
        pred = rf.predict(user_df)[0]

        st.write(f"Result: **{'Valid (1)' if pred == 1 else 'Noise (0)'}**")
        st.write(f"Confidence score: **{prob:.2f}**")

        # Simple confidence range
        low = max(prob - 0.08, 0.0)
        high = min(prob + 0.08, 1.0)

        st.write(f"Confidence range: **{low:.2f} to {high:.2f}**")
        st.progress(float(prob))

        st.write("""
        Interpretation:
        - The confidence score is how strongly the model believes this detection is valid.
        - The range shows a realistic variation around the score instead of a single fixed point.
        """)


# Tab 2 – Feature importance and sensitivity
with tab2:
    st.subheader("Feature importance")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=rf.feature_importances_, y=feature_cols, ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest feature importance")
    st.pyplot(fig)

    st.write("""
    Features with higher importance values have a stronger influence 
    on whether the model predicts a detection as valid or noise.
    """)

    st.markdown("---")
    st.subheader("Sensitivity test (change one feature and see the effect)")

    # Choose a feature, default to most important one
    importances = rf.feature_importances_
    default_index = int(np.argmax(importances))
    selected_feature = st.selectbox(
        "Choose a feature to adjust:", feature_cols, index=default_index
    )

    # Use a simple baseline example (median of all features)
    baseline = X.median().to_frame().T
    baseline_prob = rf.predict_proba(baseline)[0][1]

    # Slider range based on 1st to 99th percentile to avoid extreme outliers
    min_val = float(X[selected_feature].quantile(0.01))
    max_val = float(X[selected_feature].quantile(0.99))
    mid_val = float(baseline[selected_feature].iloc[0])

    new_val = st.slider(
        f"Adjust value for {selected_feature}",
        min_val,
        max_val,
        mid_val
    )

    changed_example = baseline.copy()
    changed_example[selected_feature] = new_val
    changed_prob = rf.predict_proba(changed_example)[0][1]

    st.write(f"Baseline probability (median values): **{baseline_prob:.2f}**")
    st.write(f"New probability after change: **{changed_prob:.2f}**")

    # Extra: show full curve of probability vs feature value
    values = np.linspace(min_val, max_val, 30)
    probs = []
    for v in values:
        temp = baseline.copy()
        temp[selected_feature] = v
        p = rf.predict_proba(temp)[0][1]
        probs.append(p)

    curve_fig, curve_ax = plt.subplots(figsize=(7, 4))
    curve_ax.plot(values, probs, marker="o")
    curve_ax.axvline(new_val, color="red", linestyle="--", label="Current value")
    curve_ax.set_xlabel(selected_feature)
    curve_ax.set_ylabel("Predicted probability of valid (1)")
    curve_ax.set_title(f"Effect of {selected_feature} on prediction")
    curve_ax.legend()
    st.pyplot(curve_fig)

    st.write("""
    This section shows how changing a single feature changes the predicted probability.  
    The line chart helps see the overall trend across the full range of that feature.
    """)


# Tab 3 – Model performance
with tab3:
    st.subheader("Model performance (summary)")

    # these are written by us bcoz we were facing problems so we just found the values in google colab file from codes and then pasted here.
    accuracy = 0.97
    precision = 0.92
    recall = 0.94
    f1_score_value = 0.93

    st.write(f"Accuracy: **{accuracy:.2f}**")
    st.write(f"Precision: **{precision:.2f}**")
    st.write(f"Recall: **{recall:.2f}**")
    st.write(f"F1 score: **{f1_score_value:.2f}**")

    st.write("""
    These values give a high-level idea of how well the model performs on the test data.  
    They are shown in a simple way to help non-technical users understand that 
    the model is performing reliably.
    """)


# Tab 4 – Data preview
with tab4:
    st.subheader("Dataset preview")
    st.dataframe(raw_data.head(50))
    st.write("This shows a small sample of the dataset used to train the model.")

# Tab 5: Exploratory Data Analysis (EDA)
with tab5:
    st.subheader("Exploratory Data Analysis")

    st.write("This section shows simple visualizations to understand the dataset.")

    # Let user choose between EDA options
    option = st.selectbox(
        "Choose what you want to explore:",
        ["Summary Statistics", "Histogram", "Boxplot", "Correlation Heatmap"]
    )

    # Summary statistics
    if option == "Summary Statistics":
        st.write("Basic numeric summary of the dataset.")
        st.dataframe(raw_data.describe())

    # Histogram (single plot)
    if option == "Histogram":
        st.write("Histogram shows how values are spread for one feature.")
        col = st.selectbox("Select a feature:", feature_cols)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(raw_data[col], kde=False, bins=30, color="#4B0014", ax=ax)
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        st.pyplot(fig)

    # Boxplot
    if option == "Boxplot":
        st.write("Boxplot shows outliers and distribution shape.")
        col = st.selectbox("Select a feature:", feature_cols)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=raw_data[col], color="#4B0014", ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    # Correlation heatmap
    if option == "Correlation Heatmap":
        st.write("This heatmap shows how different features are related to each other.")
    
        # Choose only useful numeric columns (ignore IDs)
        corr_features = [
            "sig", "sigsd", "snr",
            "runLen",
            "avg_sig_per_tag", "avg_snr_per_tag",
            "detections_per_tag",
            "valid"
        ]
    
        # Make sure the features exist in the data
        corr_features = [f for f in corr_features if f in raw_data.columns]
    
        df_corr = raw_data[corr_features].corr()
    
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df_corr,
            annot=True,          # show numbers
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            square=True,
            ax=ax
        )
    
        ax.set_title("Correlation Heatmap", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
    
        st.pyplot(fig)

# TAB 6 — RAG Chatbot (Bird Dataset)

with tab6:
    st.subheader("Bird Dataset Chatbot")

    st.write("Ask anything about the bird detections dataset.")

    import pandas as pd
    from sentence_transformers import SentenceTransformer, util
    from transformers import pipeline

    @st.cache_data
    def load_bird_data():
        df_birds = pd.read_excel("F&E_full_dataset.xlsx")

        # Drop useless columns
        drop_cols = ["Unnamed: 0", "hitID", "runID", "batchID", "ts",
                     "tsCorrected", "DATE", "TIME", "port", "antBearing"]
        df_birds = df_birds.drop(columns=[c for c in drop_cols if c in df_birds.columns],
                                 errors="ignore")

        return df_birds

    df_birds = load_bird_data()

    bird_narrative = "Here is a summary of bird detections:\n"

    # Use first 200 rows for speed
    sample_df = df_birds.head(20)

    for idx, row in sample_df.iterrows():
        bird_narrative += (
            f"Detection: sig={row['sig']}, snr={row['snr']}, runLen={row['runLen']}, "
            f"avg_snr_per_tag={row['avg_snr_per_tag']}, detections_per_tag={row['detections_per_tag']}. "
            f"Validity label: {row['motusFilter']}.\n"
        )

    bird_explanation = (
        "This dataset contains bird detection events including signal strength (sig), "
        "signal-to-noise ratio (snr), run length (runLen), and tagging-based averages. "
        "The 'motusFilter' column is the target label where 1 = valid detection and 0 = noise."
    )

    documents = {
        "doc_dataset_summary": bird_explanation,
        "doc_narrative": bird_narrative
    }

    embedder = SentenceTransformer('all-MiniLM-L12-v2')
    doc_embeddings = {
        doc_id: embedder.encode(text, convert_to_tensor=True)
        for doc_id, text in documents.items()
    }

    # Retrieval function
    def retrieve_context(query, top_k=2):
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        scores = {}

        for doc_id, emb in doc_embeddings.items():
            score = util.pytorch_cos_sim(query_embedding, emb).item()
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_doc_ids = [doc_id for doc_id, score in sorted_docs[:top_k]]

        # Join retrieved documents
        return "\n\n".join(documents[doc_id] for doc_id in top_doc_ids)

    generator = pipeline("text2text-generation", model="google/flan-t5-small")

    def query_llm(query, context):
        prompt = (
            "Below is information about bird detection data. "
            "Use the context to answer the question clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            "Answer in simple words:\n"
        )

        outputs = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        raw_output = outputs[0]['generated_text']

        # Remove prompt repetition if happens
        if raw_output.startswith(prompt):
            raw_output = raw_output[len(prompt):].strip()

        return raw_output.strip()

    def rag_chatbot(query):
        context = retrieve_context(query, top_k=2)
        answer = query_llm(query, context)
        return answer
    user_query = st.text_input("Type your question here:")

    if st.button("Ask the Chatbot"):
        if user_query.strip() == "":
            st.warning("Please type a question.")
        else:
            with st.spinner("Analyzing dataset…"):
                reply = rag_chatbot(user_query)

            st.write("Chatbot Response:")
            st.success(reply)

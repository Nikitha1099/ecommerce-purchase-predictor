import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="Credit Card Default ML App", page_icon="💳", layout="wide")

st.title(" Credit Card Default Prediction & Model Evaluation Dashboard")
st.markdown("This interactive web application demonstrates the end-to-end Machine Learning deployment workflow using the **Default of Credit Card Clients** dataset.")

# Sidebar
st.sidebar.header("Configuration Panel")

model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

selected_model_name = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))[cite: 1]

st.sidebar.subheader("Test Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])[cite: 1]

@st.cache_resource
def load_artifacts(model_filename):
    model_path = os.path.join("model", model_filename)
    scaler_path = os.path.join("model", "scaler.pkl")
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler

model_filename = model_options[selected_model_name]
model, scaler = load_artifacts(model_filename)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Test dataset successfully loaded!")
    
    if "target" in test_df.columns:
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]
    else:
        st.error("Uploaded CSV must contain the 'target' column.")
        st.stop()
        
    if selected_model_name in ["Logistic Regression", "kNN"]:
        X_test_processed = scaler.transform(X_test) if scaler is not None else X_test
    else:
        X_test_processed = X_test

    if model is not None:
        y_pred = model.predict(X_test_processed)
        try:
            y_prob = model.predict_proba(X_test_processed)[:, 1]
        except Exception:
            y_prob = np.zeros_like(y_pred, dtype=float)

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.markdown(f"### Evaluation Results for: **{selected_model_name}**")[cite: 1]
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")[cite: 1]
        col2.metric("AUC Score", f"{auc:.4f}")[cite: 1]
        col3.metric("Precision", f"{prec:.4f}")[cite: 1]
        col4.metric("Recall", f"{rec:.4f}")[cite: 1]
        col5.metric("F1 Score", f"{f1:.4f}")[cite: 1]
        col6.metric("MCC Score", f"{mcc:.4f}")[cite: 1]

        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Confusion Matrix")[cite: 1]
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)
            
        with col_right:
            st.subheader("Classification Report")[cite: 1]
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format(formatter="{:.4f}"))
            
        with st.expander("View Test Data Preview"):
            st.dataframe(test_df.head(50))
    else:
        st.error(f"Model file for {selected_model_name} not found in model/ folder.")
else:
    st.info(" Please upload your `test_data.csv` file using the sidebar to view metrics and evaluation results.")[cite: 1]
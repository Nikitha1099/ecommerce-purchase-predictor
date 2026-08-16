import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Page Configuration & Header
# ==========================================
st.set_page_config(
    page_title="E-Commerce ML Dashboard", 
    page_icon="🛒", 
    layout="wide"
)

st.title("🛒 E-Commerce Customer Purchase Intention & Model Evaluation Dashboard")

# Collapsed expander with shortened description
with st.expander("ℹ️ About This Project & Dataset", expanded=False):
    st.markdown("""
    * **Overview:** Binary classification predicting customer purchase intent (`Revenue`: `1` = purchase, `0` = no purchase) based on browsing telemetry.
    * **Dataset:** UCI Online Shoppers Purchasing Intention Dataset.
    * **Models & Metrics:** Benchmarks 5 classifiers across 6 key evaluation metrics.
    * **Data Input:** Automatically loads the pipeline-generated `test_data.csv` by default, or allows custom test dataset uploads anytime using the sidebar widget.
    """)

st.markdown("---")

# ==========================================
# 2. Sidebar: Model Selection & Data Upload
# ==========================================
st.sidebar.header("⚙️ Evaluation Settings")

model_choice = st.sidebar.selectbox(
    "Select a Machine Learning Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
)

st.sidebar.markdown("### 📁 Upload Custom Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (Optional)", type=["csv"])

# ==========================================
# 3. Data Loading Logic
# ==========================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success("✅ Custom dataset loaded successfully!")
    else:
        df = load_data("test_data.csv")
        st.sidebar.info("ℹ️ Using default generated test dataset.")
except FileNotFoundError:
    st.error("🚨 `test_data.csv` not found. Please upload a dataset or run `train.py`.")
    st.stop()

# Case-insensitive target column check
revenue_cols = [col for col in df.columns if col.lower() == 'revenue']

if not revenue_cols:
    st.error("🚨 The dataset must contain a 'Revenue' column as the target variable.")
    st.stop()

target_col = revenue_cols[0]
X_test = df.drop(columns=[target_col])
y_test = df[target_col].astype(int)

# ==========================================
# 4. Model Loading Logic
# ==========================================
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

@st.cache_resource
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_pickle(model_files[model_choice])
    
    if model_choice in ["Logistic Regression", "kNN"]:
        scaler = load_pickle("model/scaler.pkl")
        X_test_processed = scaler.transform(X_test)
    else:
        X_test_processed = X_test

except FileNotFoundError as e:
    st.error(f"🚨 Model file missing: {e}. Ensure `train.py` has run.")
    st.stop()

# ==========================================
# 5. Predictions & Metrics Calculations
# ==========================================
y_pred = model.predict(X_test_processed)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
else:
    auc = roc_auc_score(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# ==========================================
# 6. Dashboard Display
# ==========================================
st.subheader(f"Evaluation Results for: {model_choice}")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{precision:.4f}")
col4.metric("Recall", f"{recall:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

st.markdown("---")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

with viz_col2:
    st.markdown("### Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# ==========================================
# 7. Sample Predictions Preview
# ==========================================
st.markdown("---")
with st.expander("📄 View Sample Test Predictions", expanded=False):
    results_df = X_test.copy()
    results_df['Actual_Revenue'] = y_test.values
    results_df['Predicted_Revenue'] = y_pred
    st.dataframe(results_df.head(100), use_container_width=True)
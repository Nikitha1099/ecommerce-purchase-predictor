import os
import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

print("Fetching 'default-of-credit-card-clients' dataset from OpenML...")
credit_data = fetch_openml(name='default-of-credit-card-clients', version=1, as_frame=True)
df = credit_data.frame

# Standardize target column name
target_col = credit_data.target.name
if target_col in df.columns:
    df['target'] = df[target_col].astype(int)
    if target_col != 'target':
        df = df.drop(columns=[target_col])
else:
    df['target'] = credit_data.target.astype(int)

# Separate features and target
X = df.drop(columns=['target'])
X = pd.get_dummies(X, drop_first=True)
y = df['target']

print(f"Dataset Loaded | Instances: {X.shape[0]}, Features: {X.shape[1]}")

# Train-test split (20% test data to create test_data.csv)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test data and scaler
os.makedirs('model', exist_ok=True)
test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv('test_data.csv', index=False)
joblib.dump(scaler, 'model/scaler.pkl')

# Train and Evaluate Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():
    if name in ["Logistic Regression", "kNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    })
    
    file_name = name.lower().replace(' ', '_')
    joblib.dump(model, f"model/{file_name}.pkl")

results_df = pd.DataFrame(results)
print("\nModel Performance Comparison Table:")
print(results_df.to_markdown(index=False))
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Create model directory
os.makedirs("model", exist_ok=True)

# Load Online Shoppers Purchasing Intention Dataset from public source
url = "https://raw.githubusercontent.com/sharmaroshan/Online-Shoppers-Purchasing-Intention/master/online_shoppers_intention.csv"
df = pd.read_csv(url)

# Prepare target: Revenue (True/False -> 1/0)
df['target'] = df['Revenue'].astype(int)
df = df.drop(columns=['Revenue'])

# Encode categorical features (Month, VisitorType, Weekend)
categorical_cols = ['Month', 'VisitorType', 'Weekend']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df.drop(columns=['target'])
y = df['target']

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save test_data.csv with target column included
test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv("test_data.csv", index=False)

# Feature scaling for Logistic Regression and kNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.pkl")

# Train all 5 required models
models = {
    "logistic_regression.pkl": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree.pkl": DecisionTreeClassifier(random_state=42),
    "knn.pkl": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes.pkl": GaussianNB(),
    "random_forest.pkl": RandomForestClassifier(random_state=42)
}

for filename, model in models.items():
    if filename in ["logistic_regression.pkl", "knn.pkl"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    joblib.dump(model, os.path.join("model", filename))

print("Successfully trained all models and generated test_data.csv!")
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def main():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Create directory for serialized models
    os.makedirs("model", exist_ok=True)

    # 2. Load Dataset from UCI ML Repository
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    print("📥 Downloading Online Shoppers dataset from UCI Repository...")
    try:
        df = pd.read_csv(data_url)
    except Exception as e:
        print(f"⚠️ Could not download directly. Attempting local read: {e}")
        df = pd.read_csv("online_shoppers_intention.csv")

    # 3. Data Preprocessing
    print("🧹 Preprocessing categorical attributes & target labels...")
    # Convert boolean columns to integer (1 / 0)
    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)

    # One-Hot Encode categorical features ('Month', 'VisitorType')
    df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)

    # Separate features (X) and target (y)
    X = df.drop(columns=['Revenue'])
    y = df['Revenue']

    # 4. Train / Test Split (Stratified to handle class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. CRITICAL: Save test_data.csv WITH the target 'Revenue' column
    print("💾 Saving test_data.csv (including 'Revenue' column for app evaluation)...")
    test_df = X_test.copy()
    test_df['Revenue'] = y_test
    test_df.to_csv("test_data.csv", index=False)

    # 6. Fit and Save Feature Scaler (StandardScaler)
    print("⚙️ Fitting StandardScaler for distance/linear models...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 7. Model Training & Serialization
    models = {
        "logistic_regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "decision_tree": (DecisionTreeClassifier(random_state=42), False),
        "knn": (KNeighborsClassifier(n_neighbors=5), True),
        "naive_bayes": (GaussianNB(), False),
        "random_forest": (RandomForestClassifier(n_estimators=100, random_state=42), False)
    }

    for name, (model, requires_scaling) in models.items():
        model_title = name.replace('_', ' ').title()
        print(f"🤖 Training {model_title}...")
        
        if requires_scaling:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)

        # Save trained model to pkl
        file_path = f"model/{name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  └ Saved model to {file_path}")

    print("\n✅ Pipeline complete! All 5 models, scaler.pkl, and test_data.csv generated successfully.")

if __name__ == "__main__":
    main()
# 🛒 E-Commerce Customer Purchase Intention & Model Evaluation Dashboard

## Live Application Deployment
Access the cloud-hosted Streamlit application here:

👉 Streamlit Live App (https://ecommerce-purchase-predictor-effq2gyfyc7qlmqhtcwtmc.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning web application designed to predict e-commerce customer purchase intent (`Revenue`) based on browsing session telemetry and user behavioral metrics. This project is built to satisfy rigorous classification model deployment guidelines, tracking multi-metric performance and providing an interactive Streamlit evaluation dashboard.

---

## 🌟 Key Features
* **Automated Data Processing & Scaling:** Handles categorical encoding (Months, Visitor Types) and applies robust feature scaling (`StandardScaler`) for distance-based and linear models.
* **Multi-Model Evaluation Suite:** Integrates 5 distinct supervised classification algorithms.
* **Comprehensive Metrics Tracker:** Computes and displays **6 critical evaluation metrics** simultaneously: Accuracy, AUC Score, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).
* **Interactive Visualizations:** Features dynamic Confusion Matrix heatmaps and classification report dataframes.
* **Dual Data Loading Mode:** Automatically loads default test telemetry upon startup while offering a fully functional sidebar CSV upload widget for custom test evaluation.

---

## 📊 Dataset Overview
* **Source:** Online Shoppers Purchasing Intention Dataset (UCI Machine Learning Repository).
* **Instances:** 12,330 session records.
* **Features:** 18 multidimensional attributes comprising numeric session durations, bounce/exit rates, Google Analytics page values, special day proximity, and categorical user demographics.
* **Target Variable:** `Revenue` (Binary: `1` for completed purchase, `0` for bounced session).

---

## 🤖 Implemented Machine Learning Models

| Model | Preprocessing Requirement | Description |
| :--- | :--- | :--- |
| **Logistic Regression** | Scaled Features (`StandardScaler`) | Linear baseline model for binary classification. |
| **Decision Tree Classifier** | Raw Features | Non-linear model splitting features based on entropy/gini impurity. |
| **k-Nearest Neighbors (kNN)** | Scaled Features (`StandardScaler`) | Instance-based learning algorithm ($k=5$). |
| **Gaussian Naive Bayes** | Raw Features | Probabilistic classifier assuming feature independence. |
| **Random Forest Classifier** | Raw Features | Ensemble bagging model combining multiple decision trees. |

---

## 📈 Tracked Performance Metrics

1. **Accuracy Score:** Overall correctness of predictions.
2. **AUC-ROC Score:** Model capability to distinguish between classes across thresholds.
3. **Precision:** Accuracy of positive purchase predictions (minimizing false positives).
4. **Recall:** Sensitivity/coverage of actual purchases captured (minimizing false negatives).
5. **F1-Score:** Harmonic mean of Precision and Recall.
6. **Matthews Correlation Coefficient (MCC):** High-reliability statistical rate for binary classification quality.

---

## 🗂️ Project Repository Structure

```text
ecommerce-purchase-predictor/
│
├── model/
│   ├── logistic_regression.pkl   # Serialized Logistic Regression model
│   ├── decision_tree.pkl         # Serialized Decision Tree model
│   ├── knn.pkl                   # Serialized kNN model
│   ├── naive_bayes.pkl           # Serialized Naive Bayes model
│   ├── random_forest.pkl         # Serialized Random Forest model
│   └── scaler.pkl                # Fitted StandardScaler object
│
├── app.py                        # Streamlit web dashboard application
├── train.py                      # Training script & pipeline automation
├── requirements.txt              # Project package dependencies
├── test_data.csv                 # Default evaluation test dataset
└── README.md                     # Comprehensive project documentation
```
---

## 📈 Model Performance & Results Analysis

A comprehensive evaluation of the 5 supervised classification models was conducted using the test split of the Online Shoppers Purchasing Intention dataset. Because e-commerce datasets are inherently prone to class imbalance (significantly more browsing sessions result in non-purchases than completed checkouts), model selection prioritized robust metrics like **AUC-ROC**, **F1-Score**, and **Matthews Correlation Coefficient (MCC)** over raw Accuracy.

## 📈 Model Performance & Results Analysis

A comprehensive evaluation of the 5 supervised classification models was conducted using the test split of the Online Shoppers Purchasing Intention dataset. Because e-commerce datasets are inherently prone to class imbalance (significantly more browsing sessions result in non-purchases than completed checkouts), model selection prioritized robust metrics like **AUC-ROC**, **F1-Score**, and **Matthews Correlation Coefficient (MCC)** over raw Accuracy.

### 📊 Comparative Performance Summary

| Model | Accuracy | AUC-Score | Precision | Recall | F1-Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Classifier** | **0.8986** | **0.9254** | **0.784** | **0.721** | **0.751** | **0.692** |
| **Decision Tree Classifier** | 86.5% | 0.810 | 0.642 | 0.655 | 0.648 | 0.560 |
| **Logistic Regression** | 88.1% | 0.895 | 0.730 | 0.612 | 0.666 | 0.598 |
| **k-Nearest Neighbors (kNN)** | 85.9% | 0.802 | 0.635 | 0.580 | 0.606 | 0.509 |
| **Gaussian Naive Bayes** | 81.4% | 0.834 | 0.531 | 0.745 | 0.620 | 0.501 |

---

### 🔍 Key Insights & Findings

1. **Top Performer (Random Forest Classifier):**
   * **Why it won:** Ensemble bagging effectively captured non-linear interactions between critical telemetry features—specifically Google Analytics `PageValues`, `ExitRates`, and `ProductRelated_Duration`.
   * **Metrics Impact:** It achieved the highest balance of Precision and Recall, minimizing both false positives (wasted marketing efforts on low-intent users) and false negatives (missing potential buyers).

2. **Linear Baseline (Logistic Regression):**
   * Provided stable, interpretable results and solid baseline performance. However, because session behavior metrics have non-linear scaling bounds, it fell slightly behind Random Forest in capturing complex purchase triggers.

3. **Instance & Probabilistic Models (kNN & Naive Bayes):**
   * **kNN** suffered slightly due to the high-dimensional feature space and sensitivity to scaling variances.
   * **Gaussian Naive Bayes** displayed high Recall (capturing many actual buyers) but suffered from lower Precision due to its strict assumption of feature independence, resulting in more false positives.

4. **Handling Class Imbalance with MCC:**
   * Raw **Accuracy** can be misleadingly high if a model simply predicts the majority class (non-purchased sessions). The **Matthews Correlation Coefficient (MCC)** confirms true predictive quality across both classes, validating that the top models perform genuinely well rather than just guessing the majority outcome.

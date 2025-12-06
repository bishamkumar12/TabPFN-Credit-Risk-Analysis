# 🏦 Credit Risk Prediction using TabPFN & Balanced Baselines

A comparative study of **Foundation Models (TabPFN v2.0)** versus **Traditional Machine Learning (Random Forest, Logistic Regression)** for predicting loan defaults. This project highlights the "Accuracy Paradox" in financial data and demonstrates rigorous handling of Class Imbalance and Data Leakage.

## 🚀 Key Features
* **Foundation Model:** Utilizes **TabPFN (Prior-Data Fitted Network)**, a Transformer pre-trained on synthetic data for zero-shot classification.
* **Data Hygiene:** Strict removal of "Leakage Features" (e.g., `total_payment`) and custom sanitization of temporal data (Unix timestamp conversion).
* **Imbalance Handling:** Implementation of **Random Oversampling** for TabPFN and **Class Weighting** for Baselines to solve "Mode Collapse" (where models strictly predict the majority class).
* **Interpretability:** Feature Intelligence Dashboards comparing Non-Linear Importance (Random Forest) with Directional Coefficients (Logistic Regression).

## 📊 Model Performance Summary

| Model | Accuracy | AUC Score | Strength | Weakness |
| :--- | :--- | :--- | :--- | :--- |
| **TabPFN v2.0** | ~74% | ~0.69 | Zero-Shot Learning, Fast Setup | Requires strict balancing |
| **Random Forest** | ~65% | ~0.69 | Robust, Non-Linear Patterns | Slower inference |
| **Logistic Reg** | ~64% | ~0.69 | Highly Interpretable, Fast | Linear boundaries only |

> **Key Insight:** Initial naive models achieved 86% accuracy by predicting "No Default" for everyone (Zero Recall). Our final balanced models sacrifice raw accuracy to achieve **real utility** (Recall > 0.60), successfully flagging high-risk borrowers.

## 🛠️ Tech Stack
* **Python 3.10+**
* **Modeling:** `tabpfn`, `scikit-learn`, `imblearn`
* **Visualization:** `matplotlib`, `seaborn`, `missingno`

## 📂 Project Structure
* `TabPFN10.ipynb`: Main notebook containing the end-to-end pipeline (Preprocessing → Training → Evaluation).
* `best_credit_risk_model_artifact.pkl`: Serialized model artifact containing the model object, feature names, and metadata.
* `data/`: Directory for the dataset (ensure leakage columns are handled).

## 🔧 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install tabpfn scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
    ```
2.  **Authenticate (for TabPFN):**
    The notebook requires a Hugging Face Access Token to download TabPFN weights (free tier available).
3.  **Execute Notebook:**
    Run cells sequentially. The pipeline automatically handles stratified subsampling, date sanitization, and final metrics generation.

## 📈 Key Visualizations
* **Confusion Matrices:** Side-by-side comparison showing the reduction of False Negatives in balanced models.
* **Feature Importance:** "Consensus Dashboard" identifying **Interest Rate** and **Debt-to-Income Ratio** as the top predictors of default.
* **ROC Curves:** Demonstrating the superior ranking capability of the Random Forest model.

---
*Submitted as part of the Pattern Recognition (CCAI-312) Course Project.*

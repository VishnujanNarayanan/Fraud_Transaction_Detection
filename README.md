# Fraud Detection Project — Accredian Internship Task

## Problem Statement
The objective of this project is to build a **Fraud Detection System** using machine learning.  
We are given a dataset that contains transaction details, and the task is to classify whether a transaction is fraudulent or not.

---

## Steps
1. Load and understand the dataset.  
2. Perform Exploratory Data Analysis (EDA).  
3. Preprocess the data (handle missing values, encode, scale).  
4. Train multiple ML models.  
5. Evaluate performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
6. Select the best model and provide business insights.  

---

## Dataset
- Input file: `Fraud.csv`  
- Target variable: `isFraud` (0 = Non-Fraud, 1 = Fraud)  
- Highly imbalanced dataset:  
  - Non-Fraud ≈ **99.87%**  
  - Fraud ≈ **0.13%**

---

## Exploratory Data Analysis (EDA)

### Key Findings
- Fraud occurs mainly in **`CASH_OUT`** and **`TRANSFER`** transactions.  
- Fraudulent transactions are typically **much larger in amount**:  
  - Avg fraud ≈ **1.47M** vs Non-fraud ≈ **178K**.  
- Strong correlation between origin and destination balances:  
  - Dropped `newbalanceOrig` (perfect redundancy with `oldbalanceOrg`).  
  - Kept both `oldbalanceDest` and `newbalanceDest` (retain anomaly signals).  
- Statistical tests confirmed significance:  
  - **Chi-Square (type vs fraud):** p < 0.0001  
  - **Mann–Whitney U-Test (amount):** p < 0.0001  

---

## Data Preprocessing
Steps applied via a custom **FraudPreprocessor** class:
- Drop identifiers and leakage columns (`nameOrig`, `nameDest`, `isFlaggedFraud`).  
- One-Hot Encode transaction types.  
- Scale numeric features with `StandardScaler`.  
- Feature Engineering:
  - `diff_orig` = oldbalanceOrg – newbalanceOrig  
  - `diff_dest` = newbalanceDest – oldbalanceDest  
  - `suspicious_flag` = transferred but destination unchanged  
  - `error_flag` = invalid/negative balances  
  - Log-transform `amount`  
  - Time-based features: hour, day_of_week + cyclical encoding  

---

## Models Trained
- **Logistic Regression (baseline & engineered features)**  
- **Random Forest Classifier**  
- **XGBoost Classifier**  
- **Isolation Forest (unsupervised anomaly detection)**  

Evaluation metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## Model Performance (Highlight)

**Logistic Regression (Engineered Features):**
- Accuracy: **96.78%**  
- Recall (Fraud): **95.31%**  

**Reduced Feature Model:**  
- ROC-AUC: **0.9946** (minimal drop)  
- Dropping weak predictors (e.g., `hour`, `always_nonfraud_type`) had little effect.  

---

## Key Predictors of Fraud
1. **Transaction Type (CASH_OUT, TRANSFER)**  
2. **Suspicious Flag** (amount sent but dest balance unchanged)  
3. **Balance Differences** (`diff_orig`, `diff_dest`)  
4. **Transaction Amount (log-scaled)**  

---

## Business Insights
- Fraudsters focus on **high-value transactions**.  
- Fraud detection should prioritize monitoring of **CASH_OUT** and **TRANSFER**.  
- **Suspicious anomalies in balances** are strong fraud indicators.  
- Threshold-based rules (amount + type) combined with ML models can reduce fraud risk.  

---

## Recommendations
- Deploy **real-time monitoring** for high-risk transaction types.  
- Trigger alerts when `suspicious_flag = 1`.  
- Validate balance updates before confirming transactions.  
- Retrain models regularly with new fraud cases.  
- Use fraud scoring instead of hard classification for better risk management.  

---

## Measuring Effectiveness
- Track reduction in fraud cases post-deployment.  
- Monitor recall (catching frauds) vs false positives.  
- Perform A/B testing with live transaction streams.  

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing + training
python fraud_detection.py

# Saved objects
fraud_preprocessor.pkl   # preprocessing pipeline
trained_model.pkl        # trained ML model

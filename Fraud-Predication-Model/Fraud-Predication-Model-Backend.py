# FinPay Fraud Detection

A machine learning pipeline for detecting fraudulent financial transactions using engineered risk features and ensemble classifiers.

## 📌 Overview

This project builds a binary classification model to flag fraudulent transactions (`is_fraud`) from the FinPay transaction dataset. It combines domain-driven feature engineering (spending pattern deviations, login/device risk signals, merchant risk) with model comparison and hyperparameter tuning to produce a deployable fraud-scoring model.

## 📂 Dataset

Source file: `FinPay_Fraud_Dataset.xlsx`

Key raw columns:

| Type | Columns |
|---|---|
| Numerical | `transaction_amount`, `account_age_days`, `weekly_transaction_amount`, `avg_transaction_amount`, `merchant_risk_score`, `customer_credit_score`, `location_distance_km` |
| Categorical | `payment_method`, `merchant_category`, `city`, `country`, `device_type`, `browser`, `previous_fraud_count`, `failed_login_attempts`, `daily_transaction_count`, `is_weekend`, `is_night`, `vpn_used`, `new_device`, `new_location`, `chargeback_history` |
| Datetime | `transaction_time` |
| Identifiers (dropped before training) | `transaction_id`, `customer_id`, `merchant_id` |
| Target | `is_fraud` |

Missing values in `customer_credit_score` are imputed with the column mean; no other nulls were present.

## 🛠️ Feature Engineering

Custom risk features derived from the raw columns:

- **Spending deviation** — `Monthly_Transaction_Amount`, `Monthly_Transaction_Amount_average`, and their ratio `Amount_Derivation`, flagged as `Highest_Red_Flag` when spending spikes well above average.
- **Behavioral risk scores** — `Risk_Score` (prior fraud × failed logins), `Location_Risk` (prior fraud + distance from usual location), `Login_Risk` (excessive failed logins).
- **Transaction anomalies** — `Largest_Transaction` (unusually large vs. average) and `Highest_Merchart_Risk` (high merchant risk score).
- **Context-based flags** — `night_vpn` (night-time + VPN use), `location_Risk` (new device + new location combo).
- **Composite flag** — `Overall_Fraud_Risk`, combining prior fraud, failed logins, new device/location, and night+VPN activity into a single high-confidence signal.
- **Time features** — `transaction_hour`, `day_of_week`, `is_weekend`, and `is_night` are (re)derived from `transaction_time`.

## 🔄 Preprocessing

- One-hot encoding (`drop_first=True`) on `city`, `country`, `device_type`, `browser`, `payment_method`, `merchant_category`.
- Binary mapping of `chargeback_history` (`Yes`/`No` → `1`/`0`).
- `StandardScaler` applied to features after train/test split (80/20, `random_state=42`).
- Identifier columns dropped prior to modeling to avoid leakage.

## 🤖 Models

| Model | Tuning |
|---|---|
| Random Forest Classifier | `GridSearchCV` over `n_estimators`: [50, 100, 200], 5-fold CV |
| K-Nearest Neighbors | `GridSearchCV` over `n_neighbors`: [3, 5, 7], 5-fold CV |

XGBoost and linear-kernel SVC were evaluated during exploration (see commented-out code) but Random Forest was carried forward as the primary model.

## 📊 Evaluation

The following are computed for the Random Forest model on the held-out test set:

- Accuracy, Precision, Recall, F1 Score
- Full `classification_report`
- Confusion matrix
- Top-15 feature importances (visualized as a bar plot)

> Run `main.py` (or the notebook) to populate actual metric values and plots — this README doesn't hardcode results since they depend on the dataset instance and random seed used at run time.

## 📦 Outputs

Trained artifacts are serialized with `joblib`:

- `model_1.pkl` — trained Random Forest model
- `scaler_1.pkl` — fitted `StandardScaler`
- `features_1.pkl` — final feature column order (needed to align new data at inference time)

## 🧰 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
openpyxl   # for reading .xlsx
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib openpyxl
```

## 🚀 Usage

1. Place `FinPay_Fraud_Dataset.xlsx` in the project directory (update the file path in the script to match your environment).
2. Run the script:
   ```bash
   python main.py
   ```
3. Trained model, scaler, and feature list are saved to the project root as `.pkl` files for reuse in inference.

## 📁 Suggested Project Structure

```
finpay-fraud-detection/
├── data/
│   └── FinPay_Fraud_Dataset.xlsx
├── main.py
├── model_1.pkl
├── scaler_1.pkl
├── features_1.pkl
├── requirements.txt
└── README.md
```

## 🔮 Next Steps

- Handle class imbalance explicitly (e.g., `class_weight='balanced'`, SMOTE) if fraud cases are rare.
- Compare Random Forest against the tuned XGBoost/SVC variants on a common metric like ROC-AUC.
- Add an inference script that loads the saved `.pkl` artifacts and scores new transactions.

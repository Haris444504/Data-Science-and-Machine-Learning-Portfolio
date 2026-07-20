import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import sheryanalysis as sh

# from sklearn.preprocessing import standardscaler

data = pd.read_excel(r"[Enter your path]")
print(data.head())
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())
print(data.shape)
data["customer_credit_score"] = data["customer_credit_score"].fillna(
    data["customer_credit_score"].mean()
)
print(data.isnull().sum())
# print(sh.analyze(data))
# -- data visualization --

# 🧱 Columns: ['transaction_id', 'customer_id', 'merchant_id', 'transaction_amount', 'transaction_time', 'payment_method', 'merchant_category', 'city', 'country', 'device_type', 'browser', 'account_age_days', 'previous_fraud_count', 'failed_login_attempts', 'daily_transaction_count', 'weekly_transaction_amount', 'avg_transaction_amount', 'merchant_risk_score', 'customer_credit_score', 'is_weekend', 'is_night', 'location_distance_km', 'vpn_used', 'new_device', 'new_location', 'chargeback_history', 'is_fraud']

# ✅ No null values found

# 🔠 Categorical Columns: ['payment_method', 'merchant_category', 'city', 'country', 'device_type', 'browser', 'previous_fraud_count', 'failed_login_attempts', 'daily_transaction_count', 'is_weekend', 'is_night', 'vpn_used', 'new_device', 'new_location', 'chargeback_history', 'is_fraud']

# 🔢 Numerical Columns: ['transaction_amount', 'account_age_days', 'weekly_transaction_amount', 'avg_transaction_amount', 'merchant_risk_score', 'customer_credit_score', 'location_distance_km']

# 📅 Datetime Columns: ['transaction_time']

# 📝 Text Columns: ['transaction_id', 'customer_id', 'merchant_id']

categorical_column = [
    "payment_method",
    "merchant_category",
    "city",
    "country",
    "device_type",
    "browser",
    "previous_fraud_count",
    "failed_login_attempts",
    "daily_transaction_count",
    "is_weekend",
    "is_night",
    "vpn_used",
    "new_device",
    "new_location",
    "chargeback_history",
    "is_fraud",
]

Numerical_column = [
    "transaction_amount",
    "account_age_days",
    "weekly_transaction_amount",
    "avg_transaction_amount",
    "merchant_risk_score",
    "customer_credit_score",
    "location_distance_km",
]


# --- Data visualization ----
for col in categorical_column:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=data)
    sns.barplot(x=col, y='is_fraud', data=data)
    plt.title(f'Countplot of {col}')
    plt.show()

for col in Numerical_column:
    plt.figure(figsize=(10, 5))
    sns.histplot(data[col],kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()


# -- Feature Engineering --
data["Monthly_Transaction_Amount"] = data["weekly_transaction_amount"] * 4
data["Monthly_Transaction_Amount_average"] = data["avg_transaction_amount"] * 4

data["Amount_Derivation"] = (
    data["Monthly_Transaction_Amount"] / data["Monthly_Transaction_Amount_average"]
)

data["Highest_Red_Flag"] = (
    data["Amount_Derivation"] > 3 * data["avg_transaction_amount"]
).astype(int)

data["Risk_Score"] = data["previous_fraud_count"] * data["failed_login_attempts"]

data["Location_Risk"] = (data["previous_fraud_count"] * 2) + data[
    "location_distance_km"
]

data["Largest_Transaction"] = (
    data["transaction_amount"] > 2 * data["avg_transaction_amount"]
).astype(int)

data["Highest_Merchart_Risk"] = (data["merchant_risk_score"] >= 8).astype(int)

data["Login_Risk"] = (data["failed_login_attempts"] > 3).astype(int)

data["night_vpn"] = ((data["is_night"] == 1) & (data["vpn_used"] == 1)).astype(int)

data["location_Risk"] = (
    (data["new_device"] == 1) & (data["new_location"] == 1)
).astype(int)

data["Overall_Fraud_Risk"] = (
    (data["previous_fraud_count"] > 0)
    & (data["failed_login_attempts"] > 0)
    & (data["new_device"] == 1)
    & (data["new_location"] == 1)
    & (data["night_vpn"] == 1)
).astype(int)

# -- Data Preprocessing --
# one hot Encoding for binary categorical columns


# -- Data Preprocessing --

# 1. Correctly create dummy columns for multi-category features
dummy_columns = [
    "city",
    "country",
    "device_type",
    "browser",
    "payment_method",
    "merchant_category",
]
data = pd.get_dummies(data, columns=dummy_columns, drop_first=True)

# 2. Fix spelling of chargeback history (it is 'chargeback_history' in your dataset profile)
data["chargeback_history"] = (
    data["chargeback_history"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)
)

# 3. Features already binary (0 or 1) do not need get_dummies
data["new_device"] = data["new_device"].astype(int)
data["new_location"] = data["new_location"].astype(int)


# 🔠 Categorical Columns: ['payment_method', 'merchant_category', 'city', 'country', 'device_type', 
# 'browser', 'previous_fraud_count', 'failed_login_attempts', 'daily_transaction_count', 'is_weekend', 'is_night', 
# 'vpn_used', 'new_device', 'new_location', 'chargeback_history', 'is_fraud']

# -- model selection --
drop_columns = ["transaction_id", "customer_id", "merchant_id"]
data = data.drop(columns=drop_columns)

data["transaction_time"] = pd.to_datetime(data["transaction_time"])
data["transaction_hour"] = data["transaction_time"].dt.hour
data["day_of_week"] = data["transaction_time"].dt.dayofweek
data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

data["is_night"] = (
    (data["transaction_hour"] >= 22) | (data["transaction_hour"] <= 5)
).astype(int)

# -- model selection --

X = data.drop(columns=["is_fraud", "transaction_time"])
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42)
# model_xgb.fit(X_train, y_train)
# y_pred = model_xgb.predict(X_test)

# print("Model Accuracy:", model_xgb.score(X_test, y_test))

# model_svc = SVC(kernel='linear', random_state=42)
# model_svc.fit(X_train, y_train)

# y_pred_svc = model_svc.predict(X_test)
# print("SVC Model Accuracy:", model_svc.score(X_test, y_test))

# -- model tunning  & ensemble learning--

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print("Random Forest Model Accuracy:", model_rf.score(X_test, y_test))

# Perform cross-validation

Classifier = GridSearchCV(
    (model_rf),
    param_grid={"n_estimators": [50, 100, 200]},
    cv=5,
    return_train_score=True,
    scoring="accuracy",
)

Classifier.fit(X_train, y_train)
results = pd.DataFrame(Classifier.cv_results_)
print(results[["param_n_estimators", "mean_test_score", "std_test_score"]])

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

print("KNN Model Accuracy:", model_knn.score(X_test, y_test))

Classifier_knn = GridSearchCV(
    (model_knn),
    param_grid={"n_neighbors": [3, 5, 7]},
    cv=5,
    return_train_score=True,
    scoring="accuracy",
)

Classifier_knn.fit(X_train, y_train)
results_knn = pd.DataFrame(Classifier_knn.cv_results_)
print(results_knn[["param_n_neighbors", "mean_test_score", "std_test_score"]])

from sklearn.metrics import classification_report, confusion_matrix

# Check Random Forest real performance
print("--- Random Forest Detailed Report ---")
print(classification_report(y_test, y_pred_rf))

# Print the Confusion Matrix to see exactly what was missed
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_rf))

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Model Evaluation")

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))

print("\nClassification Report")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))

feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": model_rf.feature_importances_}
)

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print(feature_importance.head(20))

plt.figure(figsize=(10, 8))

sns.barplot(data=feature_importance.head(15), x="Importance", y="Feature")

plt.title("Top 15 Important Features")
plt.show()

joblib.dump(X.columns.tolist(), "features_1.pkl")
joblib.dump(model_rf, "model_1.pkl")
joblib.dump(scaler, "scaler_1.pkl")
# joblib.dump(X.columns.tolist(), "features_1.pkl")

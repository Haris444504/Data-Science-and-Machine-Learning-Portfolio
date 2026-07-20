import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model_1.pkl")
scaler = joblib.load("scaler_1.pkl")
feature_names = joblib.load("features_1.pkl")

st.set_page_config(
    page_title="FinPay Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 FinPay Fraud Detection System")
st.write("Enter transaction information below.")


transaction_amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=2500.0
)

account_age_days = st.number_input(
    "Account Age (Days)",
    min_value=0,
    value=365
)

weekly_transaction_amount = st.number_input(
    "Weekly Transaction Amount",
    min_value=0.0,
    value=15000.0
)

avg_transaction_amount = st.number_input(
    "Average Transaction Amount",
    min_value=0.0,
    value=3000.0
)

merchant_risk_score = st.slider(
    "Merchant Risk Score",
    1,
    10,
    5
)

customer_credit_score = st.slider(
    "Customer Credit Score",
    300,
    900,
    650
)

location_distance_km = st.number_input(
    "Location Distance (KM)",
    min_value=0.0,
    value=20.0
)

previous_fraud_count = st.number_input(
    "Previous Fraud Count",
    min_value=0,
    value=0
)

failed_login_attempts = st.number_input(
    "Failed Login Attempts",
    min_value=0,
    value=0
)

daily_transaction_count = st.number_input(
    "Daily Transaction Count",
    min_value=0,
    value=3
)

is_weekend = st.selectbox(
    "Weekend Transaction",
    [0,1]
)

is_night = st.selectbox(
    "Night Transaction",
    [0,1]
)

vpn_used = st.selectbox(
    "VPN Used",
    [0,1]
)

new_device = st.selectbox(
    "New Device",
    [0,1]
)

new_location = st.selectbox(
    "New Location",
    [0,1]
)

chargeback_history = st.selectbox(
    "Chargeback History",
    [0,1]
)


Monthly_Transaction_Amount = weekly_transaction_amount * 4

Monthly_Transaction_Amount_average = avg_transaction_amount * 4

Amount_Derivation = (
    Monthly_Transaction_Amount /
    (Monthly_Transaction_Amount_average + 1)
)

Highest_Red_Flag = int(
    Amount_Derivation > 3
)

Risk_Score = (
    previous_fraud_count*3 +
    failed_login_attempts*2 +
    merchant_risk_score
)

Location_Risk = (
    previous_fraud_count*2 +
    location_distance_km
)

Largest_Transaction = int(
    transaction_amount >
    (2 * avg_transaction_amount)
)

Highest_Merchant_Risk = int(
    merchant_risk_score >= 8
)

Login_Risk = int(
    failed_login_attempts > 3
)

night_vpn = int(
    is_night==1 and vpn_used==1
)

location_Risk = int(
    new_device==1 and new_location==1
)

Overall_Fraud_Risk = int(
    previous_fraud_count>0 and
    failed_login_attempts>0 and
    new_device==1 and
    new_location==1 and
    night_vpn==1
)


sample = {}

for col in feature_names:
    sample[col] = 0

sample["transaction_amount"] = transaction_amount
sample["account_age_days"] = account_age_days
sample["weekly_transaction_amount"] = weekly_transaction_amount
sample["avg_transaction_amount"] = avg_transaction_amount
sample["merchant_risk_score"] = merchant_risk_score
sample["customer_credit_score"] = customer_credit_score
sample["location_distance_km"] = location_distance_km
sample["previous_fraud_count"] = previous_fraud_count
sample["failed_login_attempts"] = failed_login_attempts
sample["daily_transaction_count"] = daily_transaction_count
sample["is_weekend"] = is_weekend
sample["is_night"] = is_night
sample["vpn_used"] = vpn_used
sample["new_device"] = new_device
sample["new_location"] = new_location
sample["chargeback_history"] = chargeback_history

sample["Monthly_Transaction_Amount"] = Monthly_Transaction_Amount
sample["Monthly_Transaction_Amount_average"] = Monthly_Transaction_Amount_average
sample["Amount_Derivation"] = Amount_Derivation
sample["Highest_Red_Flag"] = Highest_Red_Flag
sample["Risk_Score"] = Risk_Score
sample["Location_Risk"] = Location_Risk
sample["Largest_Transaction"] = Largest_Transaction
sample["Highest_Merchant_Risk"] = Highest_Merchant_Risk
sample["Login_Risk"] = Login_Risk
sample["night_vpn"] = night_vpn
sample["location_Risk"] = location_Risk
sample["Overall_Fraud_Risk"] = Overall_Fraud_Risk

sample = pd.DataFrame([sample])

sample = sample[feature_names]

sample = scaler.transform(sample)


if st.button("🔍 Predict Fraud"):

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")

    st.write(f"Fraud Probability : **{probability*100:.2f}%**")

    if probability < 0.30:
        st.success("Risk Level : Low")

    elif probability < 0.70:
        st.warning("Risk Level : Medium")

    else:
        st.error("Risk Level : High")

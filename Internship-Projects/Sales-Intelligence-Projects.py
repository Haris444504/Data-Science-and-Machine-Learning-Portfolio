import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sheryanalysis as sh
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# loading data
data = pd.read_excel(r"[your path]")
print("---------------------------------Original data : -----------------------------")
print("\n")
print(data)

print(data.describe())
print(data.shape)
print(data.columns)

# EDA portion

# columns
# Index(['OrderID', 'Date', 'CustomerID', 'Product', 'Quantity', 'UnitPrice',
#        'ShippingAddress', 'PaymentMethod', 'OrderStatus', 'TrackingNumber',
#        'ItemsInCart', 'CouponCode', 'ReferralSource', 'TotalPrice'],
#       dtype='object')

data["CouponCode"] = data["CouponCode"].fillna("Not provide")
print(data.isnull().sum())

# data visualization

numerical_columns = ["UnitPrice", "TotalPrice"]
Categorical_Columns = [
    "Product",
    "Quantity",
    "PaymentMethod",
    "OrderStatus",
    "ItemsInCart",
    "CouponCode",
    "ReferralSource",
]
print(sh.analyze(data))

# 🔠 Categorical Columns: ['Product', 'Quantity', 'PaymentMethod', 'OrderStatus', 'ItemsInCart', 'CouponCode', 'ReferralSource']

# 🔢 Numerical Columns: ['UnitPrice', 'TotalPrice']

# 📅 Datetime Columns: ['Date']

for col in Categorical_Columns:
    sns.barplot(x=col, y="TotalPrice", data=data)
    plt.show()

for colms in numerical_columns:
    sns.scatterplot(x=colms, y="TotalPrice", data=data)
    plt.show()

new_columns = ["Date", "CustomerID", "ShippingAddress", "TrackingNumber", "OrderID"]

# label Encoding

data_cleaned = data.copy()
le = LabelEncoder()

for col in new_columns:
    data[col] = le.fit_transform(data[col])

# one hot encoding
data = pd.get_dummies(data, columns=["Product"], drop_first=True, dtype=int)
data = pd.get_dummies(data, columns=["PaymentMethod"], drop_first=True, dtype=int)
data = pd.get_dummies(data, columns=["OrderStatus"], drop_first=True, dtype=int)
data = pd.get_dummies(data, columns=["CouponCode"], drop_first=True, dtype=int)
data = pd.get_dummies(data, columns=["ReferralSource"], drop_first=True, dtype=int)

print(data.head())
print(data.shape)

# model fitting
X = data.drop("TotalPrice", axis=1)
y = data["TotalPrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# mean squared error

mse = mean_squared_error(y_test, y_pred)
# r2 score error
r2 = r2_score(y_test, y_pred)

# adjusted r2 error

n = X_test.shape[0]
p = X_test.shape[1]

adj_r2 = 1 - (1 - r2) * (n-1) / (n - p - 1)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
print(f"Adjusted R-Squared Score : {adj_r2}")

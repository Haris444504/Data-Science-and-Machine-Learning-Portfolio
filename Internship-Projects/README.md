# 🛒 Ecommerce Predictive Analytics Projects 

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Type](https://img.shields.io/badge/Type-Internship%20Projects-blueviolet)

---

## 📌 Overview

This repository contains data analysis and machine learning projects built during my internship, focused on ecommerce order data. Both projects follow the full data science pipeline — from raw data ingestion and exploratory analysis to preprocessing, feature engineering, and predictive modeling — targeting **TotalPrice prediction** from order-level features.

---

## 🗂️ Projects

---

### 📁 Project 1 — Ecommerce Order Analysis with Random Forest

**File:** `Project1_Data_Analysis.py`

An end-to-end ML pipeline on an ecommerce orders dataset, using a **Random Forest Regressor** to predict total order price.

#### 🔍 Key Steps
| Step | Details |
|------|---------|
| Data Loading | Excel dataset loaded via Pandas |
| Missing Value Handling | `CouponCode` null values filled with `"No Coupon"` |
| Deduplication | Removed duplicate rows |
| Correlation Analysis | Heatmap generated on numerical features |
| Feature Engineering | One-Hot Encoding on 5 categorical columns |
| Leakage Prevention | Dropped `UnitPrice`, `Quantity`, and identifier columns to prevent data leakage |
| Modeling | `RandomForestRegressor` (100 estimators) |
| Evaluation | MSE and R² Score |

#### 🧠 Model
```
Algorithm     : Random Forest Regressor
Estimators    : 100
Test Size     : 20%
Target        : TotalPrice
```

---

### 📁 Project 3 — Ecommerce Order Analysis with Linear Regression

**File:** `Project3_Data_Analysis.py`

A comparative analysis using **Linear Regression** on the same domain, with extended evaluation metrics including **Adjusted R²**.

#### 🔍 Key Steps
| Step | Details |
|------|---------|
| Data Loading | Excel dataset loaded via Pandas |
| Missing Value Handling | `CouponCode` filled with `"Not provided"` |
| EDA | Descriptive stats, shape, column types |
| Encoding | Label Encoding for ID/date columns + One-Hot Encoding for categoricals |
| Modeling | `LinearRegression` from Scikit-learn |
| Evaluation | MSE, R² Score, and Adjusted R² Score |

#### 🧠 Model
```
Algorithm     : Linear Regression
Test Size     : 20%
Target        : TotalPrice
Metrics       : MSE | R² | Adjusted R²
```

---

## ⚙️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | Preprocessing, modeling, evaluation |
| `sheryanalysis` *(optional)* | Quick EDA summary (gracefully skipped if not installed) |

---

## 📊 Dataset Features

Both projects use an ecommerce orders dataset with the following columns:

| Column | Type | Notes |
|--------|------|-------|
| `OrderID` | Identifier | Dropped before modeling |
| `Date` | DateTime | Dropped / Label Encoded |
| `CustomerID` | Identifier | Dropped before modeling |
| `Product` | Categorical | One-Hot Encoded |
| `Quantity` | Categorical | Used as feature |
| `UnitPrice` | Numerical | Dropped (data leakage risk) |
| `ShippingAddress` | Identifier | Dropped before modeling |
| `PaymentMethod` | Categorical | One-Hot Encoded |
| `OrderStatus` | Categorical | One-Hot Encoded |
| `TrackingNumber` | Identifier | Dropped before modeling |
| `ItemsInCart` | Categorical | Used as feature |
| `CouponCode` | Categorical | Null-filled + One-Hot Encoded |
| `ReferralSource` | Categorical | One-Hot Encoded |
| `TotalPrice` | Numerical | **Target Variable** |

---

## 📁 Repository Structure

```
Ecommerce-Predictive-Analytics/
│
├── Project1_Data_Analysis.py       # Random Forest pipeline
├── Project3_Data_Analysis.py       # Linear Regression pipeline
├── datasets/
│   ├── Dataset for Data Analytics.xlsx
│   └── Data_Set_one.xlsx
└── README.md
```

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Ecommerce-Predictive-Analytics.git
   cd Ecommerce-Predictive-Analytics
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
   ```

3. **Update dataset path** in each script:
   ```python
   data = pd.read_excel("datasets/your_file.xlsx")
   ```

4. **Run a project**
   ```bash
   python Project1_Data_Analysis.py
   python Project3_Data_Analysis.py
   ```

---

## 💡 Key Learnings

- Identified and handled **data leakage** by dropping columns that directly compute the target
- Applied both **Label Encoding** and **One-Hot Encoding** based on column semantics
- Compared tree-based vs linear models on the same dataset
- Calculated **Adjusted R²** for a more honest model performance measure
- Handled optional third-party modules gracefully without breaking the pipeline

---

## 👤 Author

**Haris Saddique**
BS Computer Science — University of Management and Technology (UMT), Lahore
Founder, FLozenAI | AI Automation & Education

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/your-username)

---

## 📄 License

This repository is part of internship work and is shared for learning and portfolio purposes only.

# 📊 Data Science & Machine Learning Projects

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-76b900)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Projects](https://img.shields.io/badge/Projects-6-blueviolet)

> A collection of end-to-end Data Science and Machine Learning projects covering multiple real-world domains — ecommerce, healthcare, automotive, insurance, and workforce analytics.

---

## 🗂️ Table of Contents

- [Project Overview](#-project-overview)
- [Project 1 — eCommerce Data Analysis](#-project-1--ecommerce-data-analysis)
- [Project 2 — IT Company Workforce Analytics](#-project-2--it-company-workforce-analytics)
- [Project 3 — Patient Data Analysis](#-project-3--patient-data-analysis)
- [Project 4 — Heart Disease Prediction](#-project-4--heart-disease-prediction)
- [Project 5 — Car Price Prediction](#-project-5--car-price-prediction)
- [Project 6 — Insurance Charges Prediction](#-project-6--insurance-charges-prediction)
- [Tech Stack](#️-tech-stack)
- [Skills Demonstrated](#-skills-demonstrated)
- [How to Run](#️-how-to-run)
- [Repository Structure](#-repository-structure)
- [Author](#-author)

---

## 📋 Project Overview

| # | Project | Domain | Type | Algorithm |
|---|---------|--------|------|-----------|
| 1 | eCommerce Data Analysis | Retail | EDA + Business Intelligence | Statistical Analysis |
| 2 | IT Company Workforce Analytics | HR / Tech | Data Cleaning + Feature Engineering | Domain-based Imputation |
| 3 | Patient Data Analysis | Healthcare | EDA + Business Intelligence | Statistical Analysis |
| 4 | Heart Disease Prediction | Healthcare | Classification + PCA | Scikit-learn + Dimensionality Reduction |
| 5 | Car Price Prediction | Automotive | Regression | Linear Regression |
| 6 | Insurance Charges Prediction | Finance | Regression + Feature Selection | Linear Regression + Chi² / Pearson |

---

## 🛒 Project 1 — eCommerce Data Analysis

**File:** `eCommerce_Data_Analysis.py`

End-to-end exploratory data analysis on a customer orders dataset, extracting actionable business intelligence about revenue, customer behavior, and delivery performance.

### 🔍 Key Steps

| Step | Details |
|------|---------|
| Data Cleaning | Removed duplicate orders by `Order_ID`, handled nulls |
| Feature Engineering | Created `Total_Spend = Quantity × Price`, extracted `Year`, `Month`, `Day` from `Order_Date` |
| Business Questions | Total revenue, AOV, most popular categories, top city by revenue, late delivery % |
| Visualizations | Bar chart (product categories), pie chart (payment methods), histogram (age), heatmap (Age/Qty/Spend), boxplot (spending by gender) |

### 📊 Business Insights Extracted
- Total Revenue & Average Order Value (AOV)
- Most popular product categories by sales count
- Top revenue-generating city
- Customer age group spending distribution
- Late delivery percentage calculation

**Libraries:** `pandas` `numpy` `seaborn` `matplotlib` `scipy`

---

## 🏢 Project 2 — IT Company Workforce Analytics

**File:** `IT_COMPANY_DATA.PY`

Advanced data cleaning and feature engineering on a **corrupted** IT company dataset — the most technically complex project in the repo. Missing values in `Salary`, `Performance_Score`, `Experience_Years`, and `Projects` were imputed using **domain-based range logic** rather than simple mean/median.

### 🔍 Key Steps

| Step | Details |
|------|---------|
| Smart Imputation | Missing salary/score/experience/projects filled using conditional ranges based on correlated columns |
| Rank Feature Engineering | Created `Rank` column (`Low / Average / High`) using `np.select` with multi-condition logic |
| Top Performers | Extracted top 5 records by salary, experience, performance score, and project count |
| Visualizations | Salary distribution (`displot`), Salary vs Experience joint regression plot |
| Export | Cleaned dataset exported to Excel |

### 💡 Notable Technique
This project avoids naive mean imputation — instead, it infers missing values by cross-referencing multiple correlated columns (e.g., if salary is high → infer high experience), demonstrating real-world data cleaning thinking.

**Libraries:** `pandas` `numpy` `matplotlib` `seaborn`

---

## 🏥 Project 3 — Patient Data Analysis

**File:** `Patient_data_Analysis.py`

Healthcare analytics pipeline on patient appointment data to understand visit trends, revenue distribution, and doctor performance.

### 🔍 Key Steps

| Step | Details |
|------|---------|
| Data Cleaning | Removed duplicate patient records by `Patient_ID` |
| Datetime Engineering | Extracted `Year`, `Month`, `Day` from `Appointment_Date` |
| Business Questions | Total consultation revenue, busiest department, top doctor, most common age group, no-show % |
| Visualizations | Bar chart (patients per dept), pie chart (payment methods), heatmap (Age/Fees/Visit status), boxplot (fees by dept) |

### 📊 Business Insights Extracted
- Total consultation revenue
- Highest patient volume department
- Doctor handling the most patients
- No-show appointment percentage
- Highest revenue-generating city

**Libraries:** `pandas` `numpy` `seaborn` `matplotlib` `scipy`

---

## ❤️ Project 4 — Heart Disease Prediction

**File:** `Heart_disease_Prediction.py`

Binary classification pipeline to predict heart disease presence based on clinical features, with domain-specific preprocessing and PCA-based dimensionality reduction.

### 🔍 Key Steps

| Step | Details |
|------|---------|
| Domain Preprocessing | Cholesterol and RestingBP zero values replaced with column means (medically invalid zeros) |
| EDA Insights | Age group heart attack rates, sex-based risk, chest pain type analysis, exercise angina impact |
| Encoding | One-Hot Encoding for all categorical features |
| Scaling | `StandardScaler` on numerical columns (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) |
| Dimensionality Reduction | PCA with `n_components=0.95` — retained 95% variance |

### 📊 Domain Insights from EDA
| Feature | Finding |
|---------|---------|
| Age (Elderly) | 74.4% heart disease rate |
| Sex (Male) | 64.9% vs Female 22.2% |
| Exercise Angina (Yes) | 85.7% heart disease rate |
| Fasting Blood Sugar (High) | 80% heart disease rate |
| Asymptomatic Chest Pain | 78.4% heart disease rate |

**Libraries:** `pandas` `numpy` `matplotlib` `seaborn` `scikit-learn` `sheryanalysis`

---

## 🚗 Project 5 — Car Price Prediction

**File:** `Car_Price_Predication_model.py`

Regression pipeline on Ford car listings dataset to predict selling price. Compares two encoding strategies — One-Hot Encoding vs Label Encoding — and evaluates both approaches.

### 🔍 Key Steps

| Step | Details |
|------|---------|
| EDA | Price distribution, correlation heatmap, price vs mileage scatter, price by year/fuel/transmission boxplots |
| Encoding Strategy A | One-Hot Encoding for `model`, `transmission`, `fuelType` |
| Encoding Strategy B | Label Encoding for the same columns |
| Scaling | `StandardScaler` on numerical columns (`year`, `mileage`, `engineSize`, `tax`, `mpg`) |
| Model | `LinearRegression` trained and evaluated on both encoded versions |
| Evaluation | R² Score and Adjusted R² Score for both strategies |

### 🧠 Model
```
Algorithm     : Linear Regression
Dataset       : Ford cars CSV
Target        : price
Encoding      : One-Hot (v1) + Label Encoding (v2) — compared
Metrics       : R² | Adjusted R²
```

**Libraries:** `pandas` `numpy` `seaborn` `matplotlib` `scikit-learn`

---

## 🏦 Project 6 — Insurance Charges Prediction

**File:** `Insurance_Charges_Predictions.py`

The most statistically rigorous project — predicts medical insurance charges with advanced feature selection using **Pearson Correlation** and **Chi-Square tests** to validate which features truly matter.

### 🔍 Key Steps

| Step | Details |
|------|---------|
| EDA | Distributions for `age`, `bmi`, `children`, `charges`; outlier detection with boxplots |
| Encoding | Label Encoding for `sex` and `smoker`; One-Hot Encoding for `region` |
| Feature Engineering | `bmi_category` column created using `pd.cut` (Underweight / Normal / Overweight / Obese) |
| Scaling | `StandardScaler` on numerical columns |
| Pearson Correlation | Computed correlation of all features with `charges` target |
| Chi-Square Testing | Statistical significance test for all categorical features vs binned `charges` |
| Model | `LinearRegression` with MSE, R², and Adjusted R² |

### 📐 Statistical Feature Selection
```
Pearson Correlation  → Measures linear relationship of numerical features with charges
Chi-Square Test      → Tests independence of categorical features vs charges bins
Decision             → "Reject NULL (keep Feature)" if p-value < 0.05
```

**Libraries:** `pandas` `numpy` `seaborn` `matplotlib` `scikit-learn` `scipy`

---

## ⚙️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, cleaning |
| `numpy` | Numerical operations, conditional logic |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Preprocessing, ML models, evaluation |
| `scipy` | Statistical tests (Pearson, Chi-Square) |
| `sheryanalysis` | Quick EDA summaries (optional) |

---

## 📊 Skills Demonstrated

| Area | Topics Covered |
|------|----------------|
| **Data Wrangling** | Deduplication, null handling, type conversion, datetime parsing |
| **Smart Imputation** | Domain-based range logic (IT Company project) |
| **Feature Engineering** | `Total_Spend`, `bmi_category`, `Rank`, date components |
| **EDA** | Statistical summaries, distribution analysis, business Q&A |
| **Visualization** | Bar, pie, histogram, heatmap, boxplot, violin, displot, jointplot |
| **Encoding** | Label Encoding, One-Hot Encoding — choosing the right one per column type |
| **Scaling** | StandardScaler, MinMaxScaler |
| **Statistical Testing** | Pearson Correlation, Chi-Square test, p-value based decisions |
| **Dimensionality Reduction** | PCA (95% variance retention) |
| **ML Modeling** | Linear Regression |
| **Evaluation** | MSE, R², Adjusted R² |
| **Domain Knowledge** | Healthcare preprocessing (medically invalid zero values), HR analytics |

---

## ⚙️ How to Run

**Step 1 — Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

**Step 2 — Update the dataset path** inside any script:
```python
# Replace the hardcoded path with your own
data = pd.read_csv("your_dataset.csv")
```

**Step 3 — Run any project**
```bash
python eCommerce_Data_Analysis.py
python IT_COMPANY_DATA.PY
python Patient_data_Analysis.py
python Heart_disease_Prediction.py
python Car_Price_Predication_model.py
python Insurance_Charges_Predictions.py
```

---

## 📁 Repository Structure

```
Data-Science-and-Machine-Learning-Portfolio/
│
├── eCommerce_Data_Analysis.py          # Retail EDA & Business Intelligence
├── IT_COMPANY_DATA.PY                  # Workforce analytics + smart imputation
├── Patient_data_Analysis.py            # Healthcare appointment analytics
├── Heart_disease_Prediction.py         # Classification + PCA
├── Car_Price_Predication_model.py      # Regression + dual encoding comparison
├── Insurance_Charges_Predictions.py    # Regression + statistical feature selection
│
├── Internship-Projects/                # Separate internship work
│
└── README.md
```

---

## 👤 Author

**Haris Saddique**
BS Computer Science — University of Management and Technology (UMT), Lahore
Founder, FLozenAI | AI Automation & Education

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/haris-saddique-049189357/)
[![GitHub](https://img.shields.io/badge/GitHub-Haris444504-black?logo=github)](https://github.com/Haris444504)

---

## 📄 License

This repository is for personal learning and portfolio purposes.

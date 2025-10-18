📂 Project 1: E-commerce Data Analysis (Ecommerece_data_by_google.py)
📘 Description

Analyzes an e-commerce dataset to uncover key business insights such as:

Total revenue generated

Average order value (AOV)

Most popular product categories

Top revenue-generating cities

Age groups and payment method trends

Delivery performance (on-time vs late)

🧮 Key Steps

Data Loading & Cleaning

Remove duplicates

Handle missing values

Compute new features like Total_Spend

Extract Year, Month, and Day from order dates

Exploratory Data Analysis (EDA)

Revenue, sales, and product insights

Customer demographics

Late delivery percentage

Visualizations

📊 Bar Chart → Sales per Product Category

🥧 Pie Chart → Payment Method Distribution

📈 Line/Histogram → Age Distribution & Monthly Trends

🔥 Heatmap → Correlation between Age, Quantity, and Spend

📦 Boxplot → Spending by Gender

📦 Libraries Used
pandas, numpy, seaborn, matplotlib, scipy

📸 Example Insights

Total Revenue: Total customer spend across all orders

Top City: Highest revenue-generating location

Late Deliveries: % of orders not delivered on time


🏥 Project 2: Patient Data Analysis (Patient_data.py)
📘 Description

Performs healthcare data analytics to understand patient visit trends, fee distribution, and doctor performance.

🧮 Key Steps

Data Cleaning & Preprocessing

Convert appointment dates to datetime

Remove duplicate patient records

Add time-based columns (Year, Month, Day)

Business Insights

Total consultation revenue

Most visited department

Most common age group

Top performing doctor

No-show (missed visit) percentage

Visualizations

📊 Bar Chart → Patients per Department

🥧 Pie Chart → Payment Method Distribution

🔥 Heatmap → Correlation (Age, Fees, Visits)

📦 Boxplot → Fee Distribution by Department

📦 Libraries Used
pandas, numpy, seaborn, matplotlib, scipy

📸 Example Insights

Highest Revenue City: City with the largest total consultation fees

Top Doctor: Doctor with the highest number of patients

No-show Percentage: Patients who missed appointments

⚙️ How to Run
Prerequisites

Install all dependencies:

pip install pandas numpy matplotlib seaborn scipy

Run the Scripts
python Ecommerece_data_by_google.py
python Patient_data.py


Make sure to update the dataset paths inside the scripts before running:

pd.read_csv(r"your_path_to_dataset.csv")

🧩 Folder Structure
📁 Data-Analytics-Projects/
│
├── Ecommerece_data_by_google.py
├── Patient_data.py
├── README.md
└── datasets/
    ├── ecommerce_dataset.csv
    └── healthcare_dataset.csv

📊 Skills Demonstrated

Data Cleaning & Feature Engineering

Statistical Analysis

Data Visualization (Matplotlib, Seaborn)

Business Insight Generation

Correlation & Trend Analysis

👨‍💻 Author

Haris Saddique
Data Analytics | Machine Learning Enthusiast
📫 GitHub Profile
 | LinkedIn

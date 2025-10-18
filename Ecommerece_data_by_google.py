import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats 


plt.style.use('seaborn-v0_8')
sns.set_theme(palette= "muted",style="whitegrid")

path = pd.read_csv(r"C:\Users\Haris_hp\OneDrive\Documents\ecommerce_dataset.csv")
data = pd.DataFrame(path)

print(data.head(10))
print(data.info())
print(data.describe())

data= data.drop_duplicates(subset=['Order_ID'])

print(data.isnull().sum())
data['Total_Spend'] = data['Quantity'] * data['Price']
data['Order_Date'] = pd.to_datetime(data['Order_Date'])

data["Year"] = data['Order_Date'].dt.year
data['Month'] = data['Order_Date'].dt.month
data['Day'] = data['Order_Date'].dt.day
# Answer key business questions using Pandas + Numpy:

# What is the total revenue generated?

# Which product categories are most popular?

# What is the average order value (AOV)?

# Which age group spends the most?

# Which city/location generates the highest revenue?

# What is the distribution of payment methods?

# What % of orders were delivered late?



total_revenue = data['Total_Spend'].sum()
print("Total Revenue Generated: $", total_revenue)

AOV = total_revenue / data['Order_ID'].nunique()

print("Average Order Value (AOV) is : $" , AOV)

Most_Popular = data['Product_Category'].value_counts()
print("Most Popular Product Category is: \n", Most_Popular)

Most_Payment = data['Payment_Method'].value_counts()
print("Payment Method is: \n", Most_Payment)

Most_Popular_Age = data['Age'].value_counts()
print("Most age group spends is : " , Most_Popular_Age , "years")

city_revenue = data.groupby('Location')['Total_Spend'].sum().sort_values(ascending=False)
print("Total revenue of the cities are : " , city_revenue)

top_city = city_revenue.idxmax()
top_revenue = city_revenue.max()
print("Top revenue city : ", top_city , "with revenue of $", top_revenue)

# Late_Delivery = data['Delivery_Status'].value_counts()
# print(Late_Delivery)

Late_Deliverys = (data['Delivery_Status'] == 'Late').sum()
Late_Delivery_Percentage =  (Late_Deliverys / len(data)) * 100

print("Late Deliverys Percentage : ", Late_Delivery_Percentage , "%")

# Create resume-boosting visuals:

# Bar chart → Sales per product category

# Pie chart → Payment method distribution

# Line chart → Monthly sales trend (revenue over time)

# Histogram → Customer age distribution

# Heatmap → Correlation between Age, Quantity, Spend

# Boxplot → Distribution of spending per gender

# Stacked bar chart → On-time vs Late delivery by city


x = data['Product']
y = data['Product_Category']

sns.barplot(x = Most_Popular.index , y = Most_Popular.values , palette = 'Blues_d')
plt.xticks(rotation=45)
plt.xlabel("Product Category")
plt.ylabel("Number of Products Sold")
plt.title("Sales per Product Category")
plt.show()


plt.pie(Most_Payment , labels=Most_Payment.index , autopct='%1.1f%%' , startangle=90)
plt.title("Payment Method Distribution")
plt.show()

sns.histplot(data["Age"], kde = True , bins = 30)
plt.title("Customer Age Distribution")
plt.show()

core_data = data[["Age" , "Quantity" , "Total_Spend"]]

core_matrix = core_data.corr()
sns.heatmap(core_matrix,annot=True,cmap='coolwarm',linewidths=0.05)
plt.title("Correlation between Age, Quantity, Spend")
plt.show()

sns.boxplot(x = "Gender" , y = "Total_Spend" , data = data)
plt.title("Distribution of Spending per Gender")
plt.show()

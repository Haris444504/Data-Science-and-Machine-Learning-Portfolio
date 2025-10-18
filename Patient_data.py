import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

plt.style.use('seaborn-v0_8')

sns.set_theme(palette="muted" , style="whitegrid")

path = pd.read_csv(r"C:\Users\Haris_hp\Downloads\healthcare_dataset.csv")

data = pd.DataFrame(path)

print(data.head(10))

print(data.info())

print(data.describe())


data = data.drop_duplicates(subset=['Patient_ID'])
print(data.isnull().sum())

# managing data and times 
data["Appointment_Date"] = pd.to_datetime(data["Appointment_Date"])
data['Year'] = data['Appointment_Date'].dt.year
data['Month'] = data['Appointment_Date'].dt.month
data['day'] = data['Appointment_Date'].dt.day

# business Questions 

total_revenue = data['Fees'].sum()

print("Total revenue of consulation Fees : " , total_revenue,"$")

All_Departments = data['Department'].unique()
print(All_Departments)

Most_Patient_Deparment  = data['Department'].value_counts()
print("Hightest Number of patient departments : ",Most_Patient_Deparment.idxmax())

Average_free = total_revenue / len(data)
print("Average fees of patient is : ",Average_free,"$")

Most_Age_Group_People = data['Age'].value_counts()
print("Most age group people visited the hospital are : ",Most_Age_Group_People.idxmax())

highest_revenue_city = data.groupby('Location')['Fees'].sum().sort_values(ascending=False)
print("Highest Revenue city : ", highest_revenue_city.idxmax(), "with revenue of : ", highest_revenue_city.max(), "$")

Payment_Distribution_Method = data['Payment_Method'].value_counts()
print("Payment Distribution method is : ",Payment_Distribution_Method)

Missing_Visit = (data['Visit_Status'] == "No-show").sum()
print("Missing visit status of people are : ",Missing_Visit , "with percentage of : " , (Missing_Visit / len(data)) * 100 , "%")

Doctor_with_More_Patient = data['Doctor_Assigned'].value_counts()
print("Doctor dealing with more number of patients are : " , Doctor_with_More_Patient.idxmax())

# ----------------------------------------Data visualization------------------------------------------------


sns.barplot(x = Most_Patient_Deparment.index , y = Most_Patient_Deparment.values)
plt.title("Number of patients per department")
plt.xticks(rotation=45)
plt.xlabel("Departments")
plt.ylabel("Number of patients")
plt.show()

plt.pie(x= Payment_Distribution_Method , labels=Payment_Distribution_Method.index , autopct='%1.1f%%' , startangle=90)
plt.title("Payment method distribution")
plt.show()

data["Visiting_status_Num"] = data['Visit_Status'].map({'No-show' : 0 , 'Show' : 1})
core_data = data[["Age" , "Fees" , "Visiting_status_Num"]]
core_matrix = core_data.corr()

sns.heatmap(core_matrix , annot=True,cmap='coolwarm' , linewidths=0.05)
plt.title("Correlation (Age, Fees, Visits)")
plt.show()

sns.boxplot(x = "Fees" , y = "Department" , data=data)
plt.title("Fee distribution across departments")
plt.show()

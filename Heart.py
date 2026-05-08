import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
import sheryanalysis as sh
from sklearn.decomposition import PCA

data = pd.read_csv(r"C:\Users\Haris_hp\Downloads\heart.csv")
print(data.describe())
print(data.head(10))
print(data.columns)

print(data["HeartDisease"].value_counts())
# print(data.isnull().sum())

# sns.countplot(x="HeartDisease", data=data)
# plt.show()
# # for checking heardattack ratio with age , How age effect on heart attack
# data["Age"] = pd.qcut(data["Age"] , q=4, labels=["Young", "Middle-Aged", "Senior", "Elderly"])
# sns.countplot(x="Age", hue="HeartDisease", data=data)
# plt.show()

# result 
# Young = 36% chance of heart attack
# Middle-Aged = 47.8% chance of heart attack
# Senior = 66.6% chance of heart attack
# Elderly = 74.4% chance of heart attack

# def plotting(var, num):
#     plt.subplot(2,2,num)
#     sns.histplot(data[var] , kde=True)
#     plt.show()

# plotting("Age" , 1)
# plotting("RestingBP" , 2)
# plotting("Cholesterol" , 3)
# plotting("MaxHR" , 4)

print(data["Cholesterol"].value_counts())

# cholesterol has 172 missing values and 0 values which is not possible for cholesterol level. 
# So we will replace those values with the mean of the cholesterol column.

# Perfrorming data preprocessing

ch_mean = data.loc[data["Cholesterol"] != 0, "Cholesterol"].mean()
print(ch_mean)

bp_mean = data.loc[data["RestingBP"] != 0, "RestingBP"].mean()
data["RestingBP"] = data["RestingBP"].replace(0, bp_mean)
data["Cholesterol"] = data["Cholesterol"].replace(0, ch_mean)
data["Cholesterol"] = data["Cholesterol"].round(2)

# use the library of sheryanalysis a youtube library for data analysis and visualization 
# to check the correlation between the features and the target variable.

print(sh.analyze(data))

# sns.countplot( x = data["Sex"], hue=data["HeartDisease"])
# plt.show()
# sns.countplot( x = data["ChestPainType"], hue=data["HeartDisease"])
# plt.show()
# sns.countplot(x = data["FastingBS"], hue=data["HeartDisease"])
# plt.show()
# sns.countplot(x = data["RestingECG"], hue=data["HeartDisease"])
# plt.show()
# sns.countplot(x = data["ExerciseAngina"], hue=data["HeartDisease"])
# plt.show()

# After this analysis , Results are :
# 1. Male herat ratio = 64.9%
# 2. Female heart ratio = 22.2%

# for chest pain type
# 1. Asymptomatic = 78.4%
# 2. Non-Anginal Pain = 34.8%
# 3. Atypical Angina = 14.2%
# 4. Typical Angina = 45.45%

# with fasting blood sugar
# 1. hear rate chance = 80%
# 2. no heart rate chance = 48%

# with resting ECG
# 1. Normal = 51.8%
# 2. ST-T wave abnormality = 67.5%
# 3. Left ventricular hypertrophy = 55%

# with exercise angina
# 1. Yes = 85.7%
# 2. No = 35.1%

# sns.boxplot(x ='HeartDisease',y = 'Cholesterol' , data = data)
# plt.show()

# sns.violinplot(x='HeartDisease' , y = 'Age' , data = data)
# plt.show()

# sns.heatmap(data.corr(numeric_only=True) , annot=True)
# plt.show()

data_encode = pd.get_dummies(data , drop_first=True)
data_encode = data_encode.astype(int)
print(data_encode)

scale = StandardScaler()

numerical_Columns = ['Age' , 'RestingBP' , 'Cholesterol' , 'MaxHR'  , 'Oldpeak']

data_encode[numerical_Columns] = scale.fit_transform(data_encode[numerical_Columns])
print(data_encode)

# 1. Define your features (X) and target (y)
X = data_encode.drop('HeartDisease', axis=1)
y = data_encode['HeartDisease']

# 2. Initialize PCA
# We use n_components=0.95 to keep 95% of the information (variance)
pca = PCA(n_components=0.95)

# 3. Fit and transform the data
X_pca = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")
print(f"Variance explained by components: {pca.explained_variance_ratio_}")

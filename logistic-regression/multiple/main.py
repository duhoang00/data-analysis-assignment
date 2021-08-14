#%% - Import Data
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#%% - Load Data
df = pd.read_csv('E:/data-analysis-assignment/data-analysis-assignment/logistic-regression/multiple/bank.csv')

#%% - Validate data
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

#%% - Process data
df['default'] = df['default'].replace({"yes": 1, "no": 0})
df['loan'] = df['loan'].replace({"yes": 1, "no": 0})
df['deposit'] = df['deposit'].replace({"yes": 1, "no": 0})

#%% Transfer Data
print(df)
x = df.drop('deposit', axis='columns')
print(x)
y = df['deposit']

#%%

x['age_index'] = LabelEncoder().fit_transform(x['age'])
x['month_index'] = LabelEncoder().fit_transform(x['month'])
x['default_index'] = LabelEncoder().fit_transform(x['default'])
x['job_index'] = LabelEncoder().fit_transform(x['job'])
x['contact_index'] = LabelEncoder().fit_transform(x['contact'])
x['duration_index'] = LabelEncoder().fit_transform(x['duration'])
x['campaign_index'] = LabelEncoder().fit_transform(x['campaign'])
x['poutcome_index'] = LabelEncoder().fit_transform(x['poutcome'])
x['pdays_index'] = LabelEncoder().fit_transform(x['pdays'])
x['housing_index'] = LabelEncoder().fit_transform(x['housing'])
x['marital_index'] = LabelEncoder().fit_transform(x['marital'])
x['previous_index'] = LabelEncoder().fit_transform(x['previous'])
x['day_index'] = LabelEncoder().fit_transform(x['day'])
x['education_index'] = LabelEncoder().fit_transform(x['education'])
x['loan_index'] = LabelEncoder().fit_transform(x['loan'])
x_e=x.drop(['age','month','default','job','contact','duration','campaign','poutcome','pdays','housing','marital','previous','day','education','loan','balance'], axis='columns')
y_e = LabelEncoder().fit_transform(y)


# Create a scatter plot
plt.scatter(x, y, c=y, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()
# %%

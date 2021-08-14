#%% - Import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import statsmodels.api as sm
import seaborn as sns

#%% - Load Data
df = pd.read_csv('./bank.csv')

#%% - Validate data
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

#%% - Process data
df['default'] = df['default'].replace({"yes": 1, "no": 0})
df['loan'] = df['loan'].replace({"yes": 1, "no": 0})
df['deposit'] = df['deposit'].replace({"yes": 1, "no": 0})

print(df)

#%% Transfer Data
df['age_index'] = LabelEncoder().fit_transform(df['age'])
df['month_index'] = LabelEncoder().fit_transform(df['month'])
df['default_index'] = LabelEncoder().fit_transform(df['default'])
df['job_index'] = LabelEncoder().fit_transform(df['job'])
df['contact_index'] = LabelEncoder().fit_transform(df['contact'])
df['duration_index'] = LabelEncoder().fit_transform(df['duration'])
df['campaign_index'] = LabelEncoder().fit_transform(df['campaign'])
df['poutcome_index'] = LabelEncoder().fit_transform(df['poutcome'])
df['pdays_index'] = LabelEncoder().fit_transform(df['pdays'])
df['housing_index'] = LabelEncoder().fit_transform(df['housing'])
df['marital_index'] = LabelEncoder().fit_transform(df['marital'])
df['previous_index'] = LabelEncoder().fit_transform(df['previous'])
df['day_index'] = LabelEncoder().fit_transform(df['day'])
df['education_index'] = LabelEncoder().fit_transform(df['education'])
df['loan_index'] = LabelEncoder().fit_transform(df['loan'])
df_e=df.drop(['age','month','default','job','contact','duration','campaign','poutcome','pdays','housing','marital','previous','day','education','loan','balance'], axis='columns')

print(df_e)

#%% Checking correlation
corr = df_e.corr()
matrix = np.triu(corr)

plt.figure(figsize=(20, 10))
sns.heatmap(data=corr, annot=True, mask=matrix, cmap=plt.cm.Blues)
plt.show()

#%% Split train/test dataset
#Split Data
x = df_e.iloc[:, :-1].values
y = df_e.iloc[:, -1].values

print(x)
print(y)

#Create train/test dataset
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.35)

#%% - Create model
model = LogisticRegression(solver='lbfgs', max_iter=10000).fit( X_train, y_train)
print(model)

#%% - Get result
intercept = model.intercept_
coefs = model.coef_
score = model.score( X_train, y_train )
prob_matrix = model.predict_proba( X_train )
print(coefs)
print(intercept)

fig = plt.figure(figsize = (20, 10))
ax = plt.axes(projection ="3d")
ax.scatter3D(df['pdays'], df['age'], df['deposit'], color = "green")
plt.show()

#%% - Perform prediction using the test dataset
y_pred = model.predict( X_train )
print( classification_report( y_train, y_pred ))
cm = confusion_matrix( y_train, y_pred )

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_func(x, inter, coef):
    result = inter + coef * x
    return sigmoid(result)

#%% - Predict future values
pred_prob = prediction_func(56, intercept[0], coefs[0][0])
print(pred_prob)
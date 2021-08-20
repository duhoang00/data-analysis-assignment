#%%Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%Congigs
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 200

#%%Load data
df = pd.read_csv('./data/Insurance.csv')

#%%
plt.scatter(df.age, df['bought_insurance'], color='green', marker='o')
plt.show()

#%%Split train/test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.1)

#%%Create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)


#%%Get results
intercept = model.intercept_
coefs = model.coef_
score = model.score(X_train,y_train)
prob_matrix = model.predict_proba(X_train)
print(intercept)
print(coefs)

#%%
from sklearn.metrics import classification_report, confusion_matrix
y_predict = model.predict(X_train)
print(classification_report(y_train, y_predict))
cm = confusion_matrix(y_train,y_predict)

#%%
plt.scatter(X_train, y_train, marker='x',label='Actual')
plt.scatter(X_train, y_predict, color='red',marker='o', label='Predict')
plt.legend()
plt.show()

#%%
fig, ax = plt.subplots()
ax.imshow(cm)
ax.xaxis.set(ticks=(0,1), ticklabels=('Predict 0s', 'Predict 1s'))
ax.yaxis.set(ticks=(0,1), ticklabels=('Actual 0s', 'Actual 1s'))
for i in range(2):
  for j in range(2):
    ax.text(j, i, cm[i, j], ha ='center', va ='center', color='red', fontsize='26')

plt.show()

#%%Predict
pred_values = model.predict(X_test)
pred_score = model.score(X_test,y_test)
pred_prob_matrix = model.predict_proba(X_test)

#%%

import math
def sigmoid(x):
  return 1 / (1+math.exp(-x))

#%%Define prediction function via Sigmoid
def prediction_func(age, inter, coef):
  x = inter + coef * age
  return sigmoid(x)

#%%Draw Sigmoid Plot
plt.scatter(X_train, y_train, color='r',marker='o')
X_test = np.linspace(10,75,25)
sigs = []
for item in X_test:
  sigs.append(prediction_func(item, intercept[0], coefs[0][0]))
plt.plot(X_test, sigs, color = 'green')
plt.scatter(X_test, y_test, color='b', s=150, label='Actual')
plt.scatter(X_test, pred_values,color='orange', label='Predict')
plt.legend(loc='center right')
plt.show()

#%%Predict for future values
pred_prob = prediction_func(56, intercept[0],coefs[0][0])
print(pred_prob)

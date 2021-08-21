# %% - Import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %% - Load Data
df = pd.read_csv('./deposit_bank.csv')
print(df)

# %% - Validate data
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

# %% - Process data
df['default'] = df['default'].replace({"yes": 1, "no": 0})
df['loan'] = df['loan'].replace({"yes": 1, "no": 0})
df['deposit'] = df['deposit'].replace({"yes": 1, "no": 0})
df['housing'] = df['housing'].replace({"yes": 1, "no": 0})
df.head(10)
# %% - Exploratory data analysis
df.info()

# %% - Describe Data
df.describe()

# %% - Split train/test dataset
x = df[['default', 'loan', 'housing']]
y = df['deposit']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

# %% -Create Model
model = LogisticRegression().fit(X_train, y_train)

# %% - Get result
intercept = model.intercept_
coefs = model.coef_
score = model.score(X_train, y_train)
prob_matrix = model.predict_proba(X_train)

print(coefs)
print(intercept)
print(score)

# %% - Show Classification Report and Accuracy Score
y_pred = model.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy score:", accuracy_score(y_test, y_pred))

# %% - Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.imshow(cm)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.xaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center',
                va='center', color='white', fontsize=26)
plt.show()


# %% Visulize with ROC
y_pred_proba = model.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

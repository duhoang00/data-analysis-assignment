# %% - Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %% - Load Data
ads = pd.read_csv("./advertising.csv")
ads.head(10)

# %% - Remove NA value
ads.isnull().sum()
ads.duplicated().sum()
ads.drop_duplicates(inplace=True)

# %% - Exploratory data analysis
ads.info()

# %% Review Data
ads.describe()

# %% See Relationship effect to Click
sns.pairplot(ads, hue='Clicked on Ad')

# %% - Split train/test dataset
x = ads[['Daily Time Spent on Site', 'Age',
         'Area Income', 'Daily Internet Usage', 'Male']]
y = ads['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %% - Using SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)


# %% - Show Classification Report and Accuracy Score
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy score:", accuracy_score(y_test, y_pred))


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

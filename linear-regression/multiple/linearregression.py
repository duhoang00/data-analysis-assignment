# %% - import library
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# %% - Load data
data = pd.read_csv('./data/insurance.csv')

# %% - Check corr
data.corr()

# %% - Choose x, y
x = data[['smoker', 'bmi', 'age']]
y = data['charges']

# %% - Split into train, test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %% - Create model
model = LinearRegression().fit(X_train, y_train)

# %% - Results
R_square = model.score(X_train, y_train)
intercept = model.intercept_
coefs = model.coef_

# %% - Predict test
pred = model.predict(X_test)
predict_charges = sns.scatterplot(y_test, pred)
plt.xlabel('Actual Charges')
plt.ylabel('Predict Charges')
plt.show()

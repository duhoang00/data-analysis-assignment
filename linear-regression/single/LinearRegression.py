# %% - Import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# %% - Load data
df = pd.read_csv("./insurance.csv")

# %% - Validate data
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

# %% - Process data
df['smoker'] = df['smoker'].replace({"yes": 1, "no": 0})
df['sex'] = df['sex'].replace({"male": 1, "female": 0})

# %% - Check relationship
df.corr()
# smoker has 78.7% impact to charges
# age has 29.8% impact to charges
# bmi has 19.8% impact to charges
# children and sex has no impact to charges

# %% - See how strong of impact from smoker to charges
smoker_impact = sns.swarmplot(x='smoker', y='charges', data=df)
smoker_impact.set_title("Smoker vs Charges")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Charges")
plt.show()

# %% - Create model
x = df[['smoker']]
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# %% - Get results
R_square = model.score(x, y)
intercept = model.intercept_
slope = model.coef_

print(slope)

# %% - Predict present values
train_charges_predict = model.predict(X_train)
test_charges_predict = model.predict(X_test)

actual_predict_charges = sns.scatterplot(y_train, train_charges_predict)
actual_predict_charges.set_title("Actual Charges vs Predicted Charges")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.show()

# %% - Predict future values
future_x = [[1]]
future_y = model.predict(future_x)
print(future_y)

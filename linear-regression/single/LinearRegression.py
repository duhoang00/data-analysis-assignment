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

# %% - Check age outliers
orange_square = dict(markerfacecolor='orange', marker='s')
plt.boxplot(df['age'], vert=False, notch=True, flierprops=orange_square)
plt.xlabel("Age")
plt.title("Age Outliers")
plt.show()

# %% - Check BMI outliers
orange_square = dict(markerfacecolor='orange', marker='s')
plt.boxplot(df['bmi'], vert=False, notch=True, flierprops=orange_square)
plt.xlabel("BMI")
plt.title("BMI Outliers")
plt.show()

# %% - Check charges outliers
orange_square = dict(markerfacecolor='orange', marker='s')
plt.boxplot(df['charges'], vert=False, notch=True, flierprops=orange_square)
plt.xlabel("Charges")
plt.title("Charges Outliers")
plt.show()

# %% - Check Age and BMI outliers - DU
age_bmi = sns.scatterplot(x='age', y='bmi', hue='bmi', data=df)
plt.title('Age & BMI outliers')
plt.xlabel("Age")
plt.ylabel("BMI")
plt.show()

# %% - Age and Charges by BMI outliers - KA
age_charges_bmi = sns.scatterplot(x='age', y='charges', hue='bmi', data=df)
plt.title("Age vs Charges by BMI")
plt.legend()
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()


# %% - Age and Charges by Sex outliers - DUC
age_charges_sex = sns.scatterplot(
    x='age', y='charges', hue='sex', style='sex', data=df)
age_charges_sex.set_title("Age vs Charges by Sex")
plt.xlabel("Age (Male - 1, No - 0)")
plt.ylabel("Charges")
plt.legend()
plt.show()

# %% Age and Charges by Smoker outliers - DAT
age_charges_smoker = sns.scatterplot(
    x='age', y='charges', hue='smoker', data=df)
age_charges_smoker.set_title("Age vs Charges by Smoker")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Charges")
plt.show()

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
x = df[['age', 'bmi', 'smoker']]
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
future_x = [[24, 24.22, 1], [22, 20.98, 0], [21, 24.19, 0], [21, 21.6, 0]]
future_y = model.predict(future_x)

# %%

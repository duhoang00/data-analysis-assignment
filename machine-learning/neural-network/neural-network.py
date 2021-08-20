# %% Import lib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# %% Load data
df = pd.read_csv("./advertising.csv")
print(df)

# %% Check data types
df.info()

# %% Describe Data
df.describe()

# %% Count missing values
df.isnull().sum()

# %% Process data
x = df.drop("Clicked on Ad", axis="columns")
y = df["Clicked on Ad"]

# %% Label
x["Ad Topic Line_e"] = LabelEncoder().fit_transform(x['Ad Topic Line'])
x["City_e"] = LabelEncoder().fit_transform(x['City'])
x["Country_e"] = LabelEncoder().fit_transform(x['Country'])
x_e = x.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis='columns')
y_e = LabelEncoder().fit_transform(y)
print(x_e.head())

# %% Tranin test split
X_train, X_test, y_train, y_test = train_test_split(
    x_e, y_e, test_size=0.2, random_state=10)

# %% Create model
model = MLPClassifier(
    random_state=10, hidden_layer_sizes=41).fit(X_train, y_train)

# %% Scores, confusion_matrix, classification_report
scores = model.score(X_test, y_test)
print(scores)  # 0.81

predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

# %% Predict random value
pre_values = model.predict([[60, 21, 1, 50000, 180, 1, 91, 903]])
print(pre_values)

# %%

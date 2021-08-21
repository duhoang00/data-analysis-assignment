# %% Import lib
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
print(x)
print(y)

# %% Label
x["Ad Topic Line_e"] = LabelEncoder().fit_transform(x['Ad Topic Line'])
x["City_e"] = LabelEncoder().fit_transform(x['City'])
x["Country_e"] = LabelEncoder().fit_transform(x['Country'])
x_e = x.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis='columns')
y_e = LabelEncoder().fit_transform(y)
print(x_e.head())  # 8 cols

# %% Tranin test split
X_train, X_test, y_train, y_test = train_test_split(
    x_e, y_e, test_size=0.2, random_state=10)

# %% Create model
model = RandomForestClassifier(
    criterion='gini', random_state=10).fit(X_train, y_train)
# Show all trees
print(model.estimators_)

# %% Scores
scores = model.score(X_test, y_test)
print(scores)

# %% Plot first 5 trees
features = ["Daily Time Spent on Site",
            "Age", "Area Income", "Daily Internet Usage", "Male", "Ad Topic Line_e", "City_e", "Country_e"]
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)

for index in range(0, 5):
    plot_tree(model.estimators_[index],
              feature_names=features,
              class_names=['No', 'Yes'],
              filled=True,
              ax=axes[index])
    axes[index].set_title('Estimator: ' + str(index), fontsize=11)

fig.savefig('random-forest-first-5-trees.png')

# %% Predict random value
pre_values = model.predict([[60, 21, 1, 50000, 180, 1, 91, 903]])
print(pre_values)

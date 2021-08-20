# %% - Import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from MyFunc import sigmoid

# %% - Configs
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 200

# %% - Load data
df = pd.read_csv("./data/Insurance.csv")

# %% - Visualization
plt.scatter(df.age, df["bought_insurance"],
            color="darkgreen", marker='o')
plt.show()

# %% - Split train/test dataset
X_train, X_test, y_train, y_test = train_test_split(
    df[["age"]], df.bought_insurance, test_size=0.1)

# %% - Create model
model = LogisticRegression().fit(X_train, y_train)

# %% - Get results
intercept = model.intercept_
print(intercept)
coefs = model.coef_
print(coefs)
score = model.score(X_train, y_train)
print(score)
prob_matrix = model.predict_proba(X_train)
print(prob_matrix)

# %%
y_pred = model.predict(X_train)
print(classification_report(y_train, y_pred))
cm = confusion_matrix(y_train, y_pred)

# %%
plt.scatter(X_train, y_train, color="b", marker="o", label="Actual")
plt.scatter(X_train, y_pred, color="r", marker="+", label="Predict")
plt.legend()
plt.show()

# %%
fig, ax = plt.subplots()
ax.imshow(cm)
ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted 0s", "Predicted 1s"))
ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual 0s", "Actual 1s"))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="#FF0000", fontsize=26)
plt.show()

# %% - Predict
pred_values = model.predict(X_test)
print(pred_values)
pred_score = model.score(X_test, y_test)
print(pred_score)
pred_prob_matrix = model.predict_proba(X_test)
print(pred_prob_matrix)

# %% - Define prediction function via Sigmoid


def prediction(age, inter, coef):
    x = inter + coef * age
    return sigmoid(x)


# %% - Draw sigmoid plot
plt.scatter(X_train, y_train, color="r", marker="o")
x_test = np.linspace(10, 75, 25)
sigs = []
for item in x_test:
    sigs.append(prediction(item, intercept[0], coefs[0][0]))
plt.plot(x_test, sigs, color='g')
plt.scatter(X_test, y_test, color="b", s=150, label="Actual")
plt.scatter(X_test, pred_values, color="y", s=100, label="Predict")
plt.legend()

plt.show()

# %% - Predict future values
pred_prob = prediction(56, intercept[0], coefs[0][0])
print(pred_prob)

# %%

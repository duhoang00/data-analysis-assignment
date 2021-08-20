# %% Import Library
import pandas as pd  # for data manipulation
import numpy as np  # for data manipulation
# for building a linear regression model
from sklearn.linear_model import LogisticRegression
# for splitting the data into train and test samples
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # for model evaluation metrics
import plotly.express as px  # for data visualization
import plotly.graph_objects as go  # for data visualization

# %% Load Data set
df = pd.read_csv('./games.csv')

# %% Print the first few columns
df.iloc[:, :12]

# %% Edit new feilds and columns
df['rating_difference'] = df['white_rating']-df['black_rating']
df['white_win'] = df['winner'].apply(lambda x: 1 if x == 'white' else 0)
df['match_outcome'] = df['winner'].apply(lambda x: 1 if x == 'white' else
                                         0 if x == 'draw' else -1)

# %% Check by printing last few cols in a dataframe
df.iloc[:, 13:]

# %% Select data for modeling
X = df['rating_difference'].values.reshape(-1, 1)
y = df['white_win'].values

# %% Create training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %% Fit the model
model = LogisticRegression(solver='sag')
clf = model.fit(X_train, y_train)

# %% Predict class labels on a test data
LR1_pred_labels = model.predict(X_test)

# %% Print slope and intercept
print('Intercept (Beta 0): ', clf.intercept_)
print('Slope (Beta 1): ', clf.coef_)

# %% Use score method to get accuracy of model
score = model.score(X_test, y_test)
print('Accuracy Score: ', score)
print('--------------------------------------------------------')

# %% Look at classification report to evaluate the model
print(classification_report(y_test, LR1_pred_labels))

# %% Predict probabilities for each class label on test data
probs = model.predict_proba(X_test)

# %% Create 100 evenly spaced points from smallest X to largest X
X_range = np.linspace(X.min(), X.max(), 100)

# %% Predict probabilities for each class label
y_range = model.predict_proba(X_range.reshape(-1, 1))

# %%  Create a boolean array for masking data (needed for graph)
mask_0 = y_test < 0.5
mask_1 = y_test > 0.5
print(y_test)

# %% Create a scatter plot
fig = px.scatter(df, x=X_test.ravel(), y=y_test,
                 opacity=0.8, color_discrete_sequence=['black'],
                 labels=dict(x="Rating Points Difference Between White and Black", y="Predicted Probability for White Win",))
fig.add_traces(go.Scatter(x=X_test.ravel()[mask_1], y=probs[:, 1][mask_1]+0.01,
               name='White won', mode='markers', opacity=0.9, marker=dict(color='limegreen')))
fig.add_traces(go.Scatter(x=X_test.ravel()[mask_0], y=probs[:, 1][mask_0]-0.01,
               name='White did not win', mode='markers', opacity=0.9, marker=dict(color='red')))
fig.add_traces(go.Scatter(
    x=X_range, y=y_range[:, 1], name='Logistic Function', line=dict(color='black')))

# %% Change chart background color, Update axes lines, Set figure title
fig.update_layout(dict(plot_bgcolor='white'))
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')
fig.update_layout(title=dict(text="Binary Logistic Regression (1 Independent Variable) Model Results",
                             font=dict(color='black')))

# %% Update marker size
fig.update_traces(marker=dict(size=7))
fig.show()

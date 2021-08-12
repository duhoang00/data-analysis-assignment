#%% - import library

 import pandas as pd
 import matplotlib.pyplot as plt
 from sklearn.preprocessing import LabelEncoder
 from sklearn import tree
 from sklearn.tree import DecisionTreeClassifier

#%% - Some config

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 200

#%% - Load data

df =pd.read_csv('./data/BuyComputer.csv')
x = df.drop('Buys_computer', axis='columns')
y = df['Buys_computer']

#%%

#d = {"<=30":0,"31..40":1, ">40":2}
#df['Age'] = df['Age'].map(d)

#%%

x['Age_e'] = LabelEncoder().fit_transform(x['Age'])
x['Income_e'] = LabelEncoder().fit_transform(x['Income'])
x['Student_e'] = LabelEncoder().fit_transform(x['Student'])
x['Credit_rating_e'] = LabelEncoder().fit_transform(x['Credit_rating'])
x_e=x.drop(['Age','Income','Student','Credit_rating'], axis='columns')
y_e = LabelEncoder().fit_transform(y)

#%%

model = DecisionTreeClassifier(criterion='gini', random_state=10).fit(x_e,y_e)

#%%

features = ['Age', 'Income','Student','Credit_rating']
text_representation = tree.export_text(model, feature_names=features)
print(text_representation)

#%%

plt.figure(figsize=(20,20), dpi=200)
t = tree.plot_tree(model, feature_names=features, class_names =['No', 'Yes'], filled = True)

#%%

pre_value = model.predict([[1,1,1,1]])


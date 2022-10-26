#https://www.kaggle.com/competitions/spaceship-titanic/overview
#accuray_score:0.79

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings



data = pd.read_csv('../input/spaceship-titanic/train.csv')
#data = data.dropna(axis=0,how='any')

data1 = data.values
label = data1[:, 1]
int_label = LabelEncoder()
label = int_label.fit_transform(label)

label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()
new = pd.DataFrame(label, columns=[int_label.classes_])
new1 = new.iloc[:, :3]


data1 = data.values
label = data1[:, 2]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()
new = pd.DataFrame(label, columns=[int_label.classes_])
new2 = new.iloc[:, :2]
newdata = pd.concat([new1,new2], axis = 1) 



data1 = data.values
label = data1[:, 3]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
label = pd.DataFrame(label)

newdata = pd.concat([newdata,label], axis = 1) 



data1 = data.values
label = data1[:, 4]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()
new3 = pd.DataFrame(label, columns=[int_label.classes_])
new3 = new3.iloc[:, :3]
newdata = pd.concat([newdata,new3], axis = 1) 



new3 = data.iloc[:, 5]
newdata = pd.concat([newdata,new3], axis = 1)



data1 = data.values
label = data1[:, 6]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()
new3 = pd.DataFrame(label, columns=[int_label.classes_])
new3 = new3.iloc[:, :2]
newdata = pd.concat([newdata,new3], axis = 1) 


new3 = data.iloc[:, 7:14]
newdata = pd.concat([newdata,new3], axis = 1)
'''
newdata = newdata.dropna(axis=0,how='any')
'''
newdata.fillna(method='ffill', inplace=True)

x = newdata.iloc[:, :17]


data1 = newdata.values
label = data1[:, 18]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)

y = label.ravel()

data = pd.read_csv('../input/spaceship-titanic/test.csv')

data1 = data.values
label = data1[:, 1]
int_label = LabelEncoder()
label = int_label.fit_transform(label)

label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()
new = pd.DataFrame(label, columns=[int_label.classes_])
new1 = new.iloc[:, :3]


data1 = data.values
label = data1[:, 2]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()
new = pd.DataFrame(label, columns=[int_label.classes_])
new2 = new.iloc[:, :2]
newdata = pd.concat([new1,new2], axis = 1)


data1 = data.values
label = data1[:, 3]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
label = pd.DataFrame(label)

newdata = pd.concat([newdata,label], axis = 1) 



data1 = data.values
label = data1[:, 4]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()
new3 = pd.DataFrame(label, columns=[int_label.classes_])
new3 = new3.iloc[:, :3]
newdata = pd.concat([newdata,new3], axis = 1) 


new3 = data.iloc[:, 5]
newdata = pd.concat([newdata,new3], axis = 1)



data1 = data.values
label = data1[:, 6]
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()
new3 = pd.DataFrame(label, columns=[int_label.classes_])
new3 = new3.iloc[:, :2]
newdata = pd.concat([newdata,new3], axis = 1) 



new3 = data.iloc[:, 7:14]
newdata = pd.concat([newdata,new3], axis = 1)
newdata.fillna(method='ffill', inplace=True)

test_data = newdata.iloc[:, :17]


log = x.iloc[:, 9].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 9] = log


log = x.iloc[:, 12].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 12] = log


log = x.iloc[:, 13].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 13] = log


log = x.iloc[:, 14].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 14] = log


log = x.iloc[:, 15].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 15] = log


log = x.iloc[:, 16].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 16] = log


models = []
models.append(('LR', LogisticRegression()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('GNB', GaussianNB()))
models.append(('SVC', SVC()))

names = []
results = []

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = ("%s: %f (%f)") % (name, cv_results.mean(), cv_results.std())
    print(msg)

    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7)
lr = LogisticRegression(solver = 'newton-cg', multi_class='multinomial')


lr.fit(x_train, y_train)
pred = lr.predict(x_test)

print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

lr.fit(x, y)

parameters = [
	{
	    'multi_class': ['ovr', 'multinomial']
	}
]

model = LogisticRegression()
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
grid = GridSearchCV(estimator=model, param_grid = parameters, scoring = 'neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(x, y)

print(("Best: %f using %r") % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
  print("%f (%f) with: %r" %(mean, std, param))

x = test_data
log = x.iloc[:, 9].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 9] = log


log = x.iloc[:, 12].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 12] = log


log = x.iloc[:, 13].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 13] = log


log = x.iloc[:, 14].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 14] = log


log = x.iloc[:, 15].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 15] = log


log = x.iloc[:, 16].values.reshape(-1,1)
scaler = StandardScaler().fit(log)
log = scaler.transform(log)
x.iloc[:, 16] = log

print(x.shape)

pred = lr.predict(x)

res = list(pred)
end = []

cnt = 0
for i in res:
  if(i == 0):
    end.append((cnt, "False"))
    cnt = cnt+1
  else:
    end.append((cnt, "True"))
    cnt = cnt+1
  

xxx = pd.DataFrame(end)
xxx = xxx.iloc[:, 1]
print(xxx.head(20))





data = pd.read_csv('../input/spaceship-titanic/test.csv')
name = data.iloc[:, 0]
result = pd.concat([name, xxx], axis = 1)
so = result.iloc[:, 0:2]
print(so.shape)
print(so.head(20))
result.to_csv('sample_submission1.csv', index=0, header=["PassengerId", "Transported"])

#v2 accuracy:0.78746

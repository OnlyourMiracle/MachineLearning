#Handling Imbalanced Data with SMOTE and Near Miss Algorithm in Python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

data = pd.read_csv('/content/drive/MyDrive/MLIA/Data/creditcard.csv')

data['normAmount'] = StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis=1)
data['Class'].value_counts()
'''
print(data.info())
print(data.shape)
'''
x = data.iloc[:, 0:28]
y = data.iloc[:, -2]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
'''
print("Number transactions X_train dataset: ", x_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", x_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
'''

lr = LogisticRegression()
lr.fit(x_train, y_train.ravel())
predictions = lr.predict(x_test)
#print(classification_report(y_test, predictions))
'''
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
  '''

'''
#Using SMOTE Algorithm
sm = SMOTE(random_state = 2)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

lr.fit(x_train_res, y_train_res.ravel())
predictions = lr.predict(x_test)
'''
#print(classification_report(y_test, predictions))

#NearMiss Algorithm:
print("Before Undersampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before Undersampling, counts of label '0': {} \n".format(sum(y_train == 0)))

nr = NearMiss()
x_train_miss, y_train_miss = nr.fit_resample(x_train, y_train.ravel())

print('After Undersampling, the shape of train_X: {}'.format(x_train_miss.shape))
print('After Undersampling, the shape of train_y: {} \n'.format(y_train_miss.shape))
  
print("After Undersampling, counts of label '1': {}".format(sum(y_train_miss == 1)))
print("After Undersampling, counts of label '0': {}".format(sum(y_train_miss == 0)))

lr.fit(x_train_miss, y_train_miss.ravel())
predictions = lr.predict(x_test)
print(classification_report(y_test, predictions))

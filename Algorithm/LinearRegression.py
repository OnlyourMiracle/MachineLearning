import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import r2_score

data = pd.read_csv('/content/drive/MyDrive/Python Data analysis/insurance.csv')

nsex = data['sex']
nsmoker = data['smoker']
nloc = data['region']

int_label = LabelEncoder()
nsex = int_label.fit_transform(nsex)
nsmoker = int_label.fit_transform(nsmoker)
nloc = int_label.fit_transform(nloc)

data['sex'] = nsex
data['smoker'] = nsmoker
data['region'] = nloc


X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 1234)

lr = LinearRegression()
lr.fit(X_train, y_train)
predict = lr.predict(X_test)
r2 = r2_score(predict, y_test)

print(r2)
#r2 = 0.7

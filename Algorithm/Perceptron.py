import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn import model_selection
from sklearn.metrics import r2_score
import seaborn as sns 
from sklearn.metrics import accuracy_score


data = pd.read_csv('/content/drive/MyDrive/Python Data analysis/Social_Network_Ads.csv')
#https://www.kaggle.com/code/aysenurefe/logistic-regression

int_label = LabelEncoder()
int_sex = data['Gender']
int_sex = int_label.fit_transform(int_sex)
data['Gender'] = int_sex

X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state=1234)

sns.violinplot(data['Purchased'], data['EstimatedSalary'])
sns.despine()

'''
logreg = LogisticRegression(solver='lbfgs', max_iter=10, C=10)
logreg.fit(X_train, y_train)

result = logreg.predict(X_test)'''

clf = Perceptron(eta0 = 0.1, max_iter=100000)
clf.fit(X_train, y_train)
result = clf.predict(X_test)
print(result)
accuracy = accuracy_score(y_test, result)
print(accuracy)
#accuracy = 0.6625


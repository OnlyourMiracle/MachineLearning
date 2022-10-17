from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
dtree_model = DecisionTreeClassifier(max_depth=2).fit(x_train, y_train)
dtree_pred = dtree_model.predict(x_test)

classification_report(y_test, dtree_pred)

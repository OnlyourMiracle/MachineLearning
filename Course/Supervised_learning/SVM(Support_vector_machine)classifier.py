from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)

svm_model_linear = SVC(kernel = 'linear', C=1).fit(x_train, y_train)
svm_pred = svm_model_linear.predict(x_test)

classification_report(y_test, svm_pred)




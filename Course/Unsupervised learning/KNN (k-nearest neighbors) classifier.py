from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors = 7).fit(x_train, y_train)
knn_pred = knn.predict(x_test)

classification_report(y_test, knn_pred)




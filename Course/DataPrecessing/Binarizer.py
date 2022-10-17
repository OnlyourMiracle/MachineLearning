#We can see that all values equal or less than 0 are marked 0 and all of those above 0 are marked 1. 
import pandas
import numpy
from sklearn.preprocessing import Binarizer

array = numpy.random.random_sample((5,9)) - 0.5

print(array)
X = array[:, 0:8]
Y = array[:, 8]

binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
numpy.set_printoptions(precision=3)
print(binaryX[0:5, :])

#The values for each attribute now have a mean value of 0 and a standard deviation of 1. 

import pandas
import numpy
from sklearn.preprocessing import StandardScaler

array = numpy.random.random_sample((5,9))


X = array[:, 0:8]
Y = array[:, 8]

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
numpy.set_printoptions(precision=3)
print(rescaledX[0:5, :])

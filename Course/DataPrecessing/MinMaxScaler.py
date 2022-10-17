import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler

array = numpy.random.randint(0,100, 45)
array = array.reshape(5, 9)

X = array[:, 0:8]
Y = array[:, 8]

scaler = MinMaxScaler(feature_range=(0,1)) #Python code to Rescale data (between 0 and 1)  
rescaledX = scaler.fit_transform(X)
numpy.set_printoptions(precision=3)
print(rescaledX[0:5, :])

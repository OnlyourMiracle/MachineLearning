import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

label = label = ['male','female']  
int_label = LabelEncoder()
label = int_label.fit_transform(label)
label = np.array(label).reshape(len(label), 1)
onehot_label = OneHotEncoder()
label = onehot_label.fit_transform(label).toarray()

#convert numerical categorical variables into binary vectors
#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
  
#Retrieving data
data = pd.read_csv('Employee_data.csv')
  
# Converting type of columns to category
data['Gender']=data['Gender'].astype('category')
data['Remarks']=data['Remarks'].astype('category')
  
  
#Assigning numerical values and storing it in another columns
data['Gen_new']=data['Gender'].cat.codes
data['Rem_new']=data['Remarks'].cat.codes 
  
  
#Create an instance of One-hot-encoder
enc=OneHotEncoder()
  
#Passing encoded columns
'''
NOTE: we have converted the enc.fit_transform() method to array because the fit_transform method 
of OneHotEncoder returns SpiPy sparse matrix this enables us to save space when we 
have huge  number of categorical variables
'''
enc_data=pd.DataFrame(enc.fit_transform(data[['Gen_new','Rem_new']]).toarray())
  
#Merge with main
New_df=data.join(enc_data)
  
print(New_df)

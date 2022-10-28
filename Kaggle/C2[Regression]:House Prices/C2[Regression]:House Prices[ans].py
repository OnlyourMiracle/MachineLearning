#link:https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/
#accuracy_score=0.866

!pip install scikit-learn==0.23.2
!pip install pycaret[full]
from pycaret.regression import *  #attention:must be regression

import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import missingno as msno


    
#read data
train = pd.read_csv('/content/drive/MyDrive/MLIA/Data/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/content/drive/MyDrive/MLIA/Data/house-prices-advanced-regression-techniques/test.csv')
submission =pd.read_csv('/content/drive/MyDrive/MLIA/Data/house-prices-advanced-regression-techniques/sample_submission.csv')
all_data = pd.concat([train, test], axis=0)






#detail explore
#sme code






#imputing missing values
# Replace categorical variables with specific values (False, None) or freeest values.
print(all_data.isnull().sum())

#analy data
print(all_data.groupby('Alley').size())
print(all_data.head(20))

continuous_dist(train, 'LotFrontage', 'SalePrice')

#imputing missing values
# Replace categorical variables with specific values (False, None) or freeest values.
all_data['MSZoning'].fillna(all_data.MSZoning.mode()[0], inplace=True)
all_data['LotFrontage'].fillna('60', inplace=True)
all_data['Alley'].fillna("abc", inplace=True)
all_data['MasVnrType'].fillna("abc", inplace=True)
all_data['Utilities'].fillna(all_data.Utilities.mode()[0], inplace=True)
all_data['Exterior1st'].fillna(all_data.Exterior1st.mode()[0], inplace=True)
all_data['Exterior2nd'].fillna(all_data.Exterior2nd.mode()[0], inplace=True)
all_data['MasVnrArea'].fillna('0', inplace=True)
all_data['BsmtQual'].fillna('None', inplace=True)
all_data['BsmtCond'].fillna(all_data.BsmtCond.mode()[0], inplace=True)
all_data['BsmtExposure'].fillna(all_data.BsmtExposure.mode()[0], inplace=True)
all_data['BsmtFinType1'].fillna('None', inplace=True)
all_data['BsmtFinSF1'].fillna('0', inplace=True)
all_data['BsmtFinType2'].fillna(all_data.BsmtFinType2.mode()[0], inplace=True)
all_data['BsmtFinSF2'].fillna('0', inplace=True)
all_data['BsmtUnfSF'].fillna('0', inplace=True)
all_data['TotalBsmtSF'].fillna('0', inplace=True)
all_data['Electrical'].fillna(all_data.Electrical.mode()[0], inplace=True)
all_data['BsmtFullBath'].fillna('0', inplace=True)
all_data['BsmtHalfBath'].fillna('0', inplace=True)
all_data['KitchenQual'].fillna('None', inplace=True)
all_data['Functional'].fillna(all_data.Functional.mode()[0], inplace=True)
all_data['FireplaceQu'].fillna("abc", inplace=True)
all_data['Fence'].fillna("Fence", inplace=True)
all_data['MiscFeature'].fillna("MiscFeature", inplace=True)
all_data['SaleType'].fillna("SaleType", inplace=True)
all_data['GarageType'].fillna(all_data.GarageType.mode()[0], inplace=True)
all_data['GarageYrBlt'].fillna('None', inplace=True)
all_data['GarageFinish'].fillna(all_data.GarageFinish.mode()[0], inplace=True)  #attention
all_data['GarageCars'].fillna('0', inplace=True)
all_data['GarageArea'].fillna('0', inplace=True)
all_data['GarageQual'].fillna(all_data.GarageQual.mode()[0], inplace=True)
all_data['GarageCond'].fillna(all_data.GarageCond.mode()[0], inplace=True)





#encoding
print(all_data.dtypes)

all_data['GarageYrBlt'].replace('None', 2005, inplace=True)
print(all_data['MasVnrArea'].describe)

le = LabelEncoder()

all_data['MasVnrArea'] = le.fit_transform(all_data['MasVnrArea'])

all_data['GarageArea'] = all_data['GarageArea'].astype('int')

all_data.drop(['Id'], axis=1, inplace=True)

for col in all_data.columns[all_data.dtypes == object]:
  le = LabelEncoder()
  all_data[col] = le.fit_transform(all_data[col])

all_data.to_csv('/content/drive/MyDrive/MLIA/Data/house-prices-advanced-regression-techniques/House_all_data.csv')

#Normalize Data
from sklearn.preprocessing import StandardScaler

all_data['YearRemodAdd'] = StandardScaler().fit_transform(np.array(all_data['YearRemodAdd']).reshape(-1,1))
all_data['BsmtFinSF1'] = StandardScaler().fit_transform(np.array(all_data['BsmtFinSF1']).reshape(-1,1))
all_data['BsmtUnfSF'] = StandardScaler().fit_transform(np.array(all_data['BsmtUnfSF']).reshape(-1,1))
all_data['TotalBsmtSF'] = StandardScaler().fit_transform(np.array(all_data['TotalBsmtSF']).reshape(-1,1))
all_data['1stFlrSF'] = StandardScaler().fit_transform(np.array(all_data['1stFlrSF']).reshape(-1,1))
all_data['2ndFlrSF'] = StandardScaler().fit_transform(np.array(all_data['2ndFlrSF']).reshape(-1,1))
all_data['GrLivArea'] = StandardScaler().fit_transform(np.array(all_data['GrLivArea']).reshape(-1,1))
all_data['GarageArea'] = StandardScaler().fit_transform(np.array(all_data['GarageArea']).reshape(-1,1))
all_data['WoodDeckSF'] = StandardScaler().fit_transform(np.array(all_data['WoodDeckSF']).reshape(-1,1))
all_data['OpenPorchSF'] = StandardScaler().fit_transform(np.array(all_data['OpenPorchSF']).reshape(-1,1))
all_data['EnclosedPorch'] = StandardScaler().fit_transform(np.array(all_data['EnclosedPorch']).reshape(-1,1))
all_data['YearRemodAdd'] = StandardScaler().fit_transform(np.array(all_data['YearRemodAdd']).reshape(-1,1))


all_data.drop(['Id'], axis=1, inplace=True)
print(all_data.isnull().sum())



#split data

train, x_test = all_data.iloc[:train.shape[0]], all_data.iloc[train.shape[0]:].drop(['SalePrice'], axis=1)
x_train, y_train = train.drop(['SalePrice'], axis=1), train['SalePrice']


#modeling and optimizing
s = setup(data=train,
          session_id=7010,
          target='SalePrice',
          train_size=0.99,
          fold_strategy='stratifiedkfold',
          fold=5,
          fold_shuffle=True,
          silent=True,
          ignore_low_variance=True,
          remove_multicollinearity = True,
          normalize = True,
          normalize_method = 'robust')

top4 = compare_models()
print(top4)
print(top4.get_all_params())

!pip install scikit-optimize
!pip install tune-sklearn ray[tune]
!pip install optuna
import optuna
!pip install hpbandster ConfigSpace


#lightgbm = tune_model(create_model('catboost'), choose_better = True, n_iter = 20)

lightgbm2 = tune_model(create_model('catboost'), optimize='Accuracy', 
                        search_library='scikit-optimize', search_algorithm='bayesian', 
                        choose_better = True, n_iter = 20)
print(lightgbm2)
'''
lightgbm3 = tune_model(create_model('lightgbm'), optimize='Accuracy',
                        search_library='tune-sklearn', search_algorithm='bayesian',
                        choose_better = True, n_iter = 20)

lightgbm4 = tune_model(create_model('lightgbm'), optimize='Accuracy',
                        search_library='tune-sklearn', search_algorithm='hyperopt',
                        choose_better = True, n_iter = 20)

lightgbm5 = tune_model(create_model('lightgbm'), optimize='Accuracy',
                        search_library='tune-sklearn', search_algorithm='optuna',
                        choose_better = True, n_iter = 20)

lightgbm6 = tune_model(create_model('lightgbm'), optimize='Accuracy',
                        search_library='optuna', search_algorithm='tpe',
                        choose_better = True, n_iter = 20)
'''

catboost_best = create_model('catboost',nan_mode= 'Min', eval_metric= 'RMSE', iterations= 1000, sampling_frequency= 'PerTree', 
                             leaf_estimation_method= 'Newton', grow_policy= 'SymmetricTree', penalties_coefficient= 1, boosting_type= 'Plain',
                             model_shrink_mode= 'Constant', feature_border_type= 'GreedyLogSum',
                             eval_fraction= 0,  l2_leaf_reg= 3, random_strength= 1, rsm= 1, 
                             boost_from_average= True, model_size_reg= 0.5, subsample= 0.800000011920929, 
                             use_best_model= False,  depth= 6, posterior_sampling= False, border_count= 254,  
                              sparse_features_conflict_fraction= 0, leaf_estimation_backtracking= 'AnyImprovement', 
                             best_model_min_trees= 1, model_shrink_rate= 0, min_data_in_leaf= 1, loss_function= 'RMSE', 
                             learning_rate= 0.04339500144124031, score_function= 'Cosine',  leaf_estimation_iterations= 1, bootstrap_type= 'MVS', 
                             max_leaves= 64)

df_pred = predict_model(catboost_best, x_test)
print(df_pred.head())
y_pred = df_pred.loc[:, ['Label']]

!pip install shap
!pip install numpy==1.20
import shap
interpret_model(catboost_best)

#interpreting model
submission['SalePrice'] = y_pred
submission.to_csv('/content/drive/MyDrive/MLIA/Data/house-prices-advanced-regression-techniques/submissionEnd.csv', index=False)

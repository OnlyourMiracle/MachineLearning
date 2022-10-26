#accuracy_score:0.808

!pip install scikit-learn==0.23.2
#!pip install pycaret[full]
from pycaret.classification import *

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

PALETTE=['lightcoral', 'lightskyblue', 'gold', 'sandybrown', 'navajowhite',
        'khaki', 'lightslategrey', 'turquoise', 'rosybrown', 'thistle', 'pink']
sns.set_palette(PALETTE)
BACKCOLOR = '#f6f5f5'

from IPython.core.display import HTML

#user modules
def multi_table(table_list):
    return HTML(
        f"<table><tr> {''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list])} </tr></table>")

def cat_dist(data, var, hue, msg_show=True):
    total_cnt = data[var].count()
    f, ax = plt.subplots(1, 2, figsize=(25, 8))
    hues = [None, hue]
    titles = [f"{var}'s distribution", f"{var}'s distribution by {hue}"]

    for i in range(2):
        sns.countplot(x = data[var], edgecolor='black', hue=hues[i], linewidth=1, ax=ax[i], data=data)
        ax[i].set_xlabel(var, weight='bold', size=13)
        ax[i].set_ylabel('Count', weight='bold', size=13)
        ax[i].set_facecolor(BACKCOLOR)
        ax[i].spines[['top', 'right']].set_visible(False)
        ax[i].set_title(titles[i], size=15, weight='bold')
        for patch in ax[i].patches:
            x, height, width = patch.get_x(), patch.get_height(), patch.get_width()
            if msg_show:
                ax[i].text(x + width / 2, height + 3, f'{height} \n({height / total_cnt * 100:2.2f}%)', va='center', ha='center', size=12, bbox={'facecolor': 'white', 'boxstyle': 'round'})
    plt.show()
    
def continuous_dist(data, x, y):
    f, ax = plt.subplots(1, 4, figsize=(35, 10))
    sns.histplot(data=train, x=y, hue=x, ax=ax[0], element='step')
    sns.violinplot(x=data[x], y=data[y], ax=ax[1], edgecolor='black', linewidth=1)
    sns.boxplot(x=data[x], y=data[y], ax=ax[2])
    sns.stripplot(x=data[x], y=data[y], ax=ax[3])
    for i in range(4):
        ax[i].spines[['top','right']].set_visible(False)
        ax[i].set_xlabel(x, weight='bold', size=20)
        ax[i].set_ylabel(y, weight='bold', size=20)
        ax[i].set_facecolor(BACKCOLOR)
    f.suptitle(f"{y}'s distribution by {x}", weight='bold', size=25)
    plt.show()
    
#read data
train = pd.read_csv('/content/drive/MyDrive/MLIA/Data/spaceship-titanic/train.csv')
test = pd.read_csv('/content/drive/MyDrive/MLIA/Data/spaceship-titanic/test.csv')
submission =pd.read_csv('/content/drive/MyDrive/MLIA/Data/spaceship-titanic/sample_submission.csv')
all_data = pd.concat([train, test], axis=0)

#missing value
#msno.matrix(all_data)


#detail explore
#sme code

#imputing missing values
# Replace categorical variables with specific values (False, None) or freeest values.
all_data['CryoSleep'].fillna(False, inplace=True)
all_data['Cabin'].fillna('None', inplace=True)
all_data['VIP'].fillna(all_data.VIP.mode()[0], inplace=True)
all_data['HomePlanet'].fillna(all_data.HomePlanet.mode()[0], inplace=True)
all_data['Destination'].fillna(all_data.Destination.mode()[0], inplace=True)

# Replace continuous variables with specific values (0) or averages.
all_data['Age'].fillna(all_data.Age.mean(), inplace=True)
all_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] =\
all_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)

#create derivative variable
# As mentioned earlier, create a new variable by decomposing strings in Cabin and PassengerId.
all_data['Deck'] = all_data.Cabin.apply(lambda x:str(x)[:1])
all_data['Side'] = all_data.Cabin.apply(lambda x:str(x)[-1:])
all_data['PassengerGroup'] = all_data['PassengerId'].apply(lambda x: x.split('_')[0])
all_data['PassengerNo'] = all_data['PassengerId'].apply(lambda x: x.split('_')[1])

# Generate new variables based on the amount of money used for various services.
all_data['TotalSpend'] = all_data['RoomService'] + all_data['FoodCourt'] + all_data['ShoppingMall'] + all_data['Spa'] + all_data['VRDeck']
all_data['PctRoomService'] = all_data['RoomService']/all_data['TotalSpend']
all_data['PctFoodCourt'] = all_data['FoodCourt']/all_data['TotalSpend']
all_data['PctShoppingMall'] = all_data['ShoppingMall']/all_data['TotalSpend']
all_data['PctSpa'] = all_data['Spa']/all_data['TotalSpend']
all_data['PctVRDeck'] = all_data['VRDeck']/all_data['TotalSpend']

# Create new variables by dividing age groups.
all_data['AgeBin'] = 7
for i in range(6):
    all_data.loc[(all_data.Age >= 10*i) & (all_data.Age < 10*(i + 1)), 'AgeBin'] = i
    
# Replaces the missing value that occurred when generating the derived variable.
fill_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck']
all_data[fill_cols] = all_data[fill_cols].fillna(0)

#drop variables
# Remove unnecessary variables.
all_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)



#encoding
for col in all_data.columns[all_data.dtypes == object]:
    if col != 'Transported':
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col])
all_data['CryoSleep'] = all_data['CryoSleep'].astype('int')
all_data['VIP'] = all_data['VIP'].astype('int')

#split data

train, x_test = all_data.iloc[:train.shape[0]], all_data.iloc[train.shape[0]:].drop(['Transported'], axis=1)
x_train, y_train = train.drop(['Transported'], axis=1), train['Transported']

#modeling and optimizing
s = setup(data=train,
          session_id=7010,
          target='Transported',
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
print(top4.get_params())

!pip install scikit-optimize
!pip install tune-sklearn ray[tune]
!pip install optuna
import optuna
!pip install hpbandster ConfigSpace


#lightgbm = tune_model(create_model('catboost'), choose_better = True, n_iter = 20)

lightgbm2 = tune_model(create_model('catboost'), optimize='Accuracy', 
                        search_library='scikit-optimize', search_algorithm='bayesian', 
                        choose_better = True, n_iter = 20)
print(lightgbm2.get_all_params())
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

catboost_best = create_model('catboost',nan_mode= 'Min', eval_metric= 'Logloss', iterations=1000, sampling_frequency= 'PerTree', 
                             leaf_estimation_method= 'Newton', grow_policy= 'SymmetricTree', penalties_coefficient= 1, boosting_type= 'Plain',
                             model_shrink_mode= 'Constant', feature_border_type= 'GreedyLogSum', eval_fraction= 0,  l2_leaf_reg= 3,
                             random_strength= 1, rsm= 1, boost_from_average= False, model_size_reg= 0.5,  subsample= 0.800000011920929,
                             use_best_model= False, class_names= [0, 1], depth= 6, posterior_sampling= False, border_count= 254,
                             classes_count= 0, auto_class_weights= None, sparse_features_conflict_fraction= 0,
                             leaf_estimation_backtracking= 'AnyImprovement', best_model_min_trees= 1, model_shrink_rate= 0, 
                             min_data_in_leaf= 1, loss_function= 'Logloss', learning_rate= 0.02582800015807152, score_function= 'Cosine',
                             task_type= 'CPU', leaf_estimation_iterations= 10, bootstrap_type= 'MVS', max_leaves= 64)

df_pred = predict_model(catboost_best, x_test)
y_pred = df_pred.loc[:, ['Label']]

!pip install shap
#!pip install numpy==1.20
import shap
interpret_model(catboost_best)

#interpreting model
submission['Transported'] = y_pred
submission.to_csv('submission1.csv', index=False)


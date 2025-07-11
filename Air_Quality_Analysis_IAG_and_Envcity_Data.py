#!/usr/bin/env python
# coding: utf-8

# Standard library imports
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union
from itertools import product

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
from scipy import stats
from scipy.stats import uniform, randint

# Machine Learning
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV,
    RepeatedKFold, 
    cross_val_score,
    cross_validate
)
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error
)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.sans-serif": "Times",
    "font.size" : 10,
})

# Custom libs
try:
    from envcity_plot_lib import *
    ENVCITY_AVAILABLE = True
except ImportError:
    warnings.warn("envcity_plot_lib not available")
    ENVCITY_AVAILABLE = False

try:
    from alphasense_b_sensors.alphasense_sensors import *
    ALPHASENSE_AVAILABLE = True
except ImportError:
    warnings.warn("alphasense_b_sensors not available")
    ALPHASENSE_AVAILABLE = False

# Configuration
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.sans-serif": "Times",
    "font.size": 10,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3
})

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

def print_versions() -> None:
    """Print versions of key libraries."""
    print("="*50)
    print("LIBRARY VERSIONS")
    print("="*50)
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Matplotlib: {plt.matplotlib.__version__}")
    print(f"Seaborn: {sns.__version__}")
    print(f"Scikit-learn: {sklearn_version}")
    print("="*50)

print_versions()

def exploratory_analysis(dict_data_e1, dict_data_e2, labels, latex_labels, start, end):

    table_exploratory_analysis = {}

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l]
        e2 = dict_data_e2[l]

        concatenated = pd.concat([e1, e2], axis=1, keys=['Station 1', 'Station 2'])
        table_exploratory_analysis[l] = describe(concatenated, ['median'], ['25%', '50%', '75%'])

    return table_exploratory_analysis

class dataLoader:
    def __init__(self):
        pass
    
    

aqm = pd.read_csv('envcity_df_sp_dataset_2023.csv')

aqm.set_index('time', inplace=True)
aqm.index = pd.to_datetime(aqm.index)

labels =  ['co_we', 'co_ae', "temp"]
preffix = ['e2sp_']
label_ref= 'iag_co'

df = aqm

df = aqm[[preffix[0] + labels[0], preffix[0] + labels[1], 'pin_umid', label_ref, preffix[0] + "temp"]]

df.index = pd.to_datetime(df.index)
df = df.resample('5min').mean()
df = df.interpolate(method = 'linear', limit=1, limit_area = "inside")
df = df.dropna()

print(df.shape)

co = Alphasense_Sensors("CO-B4", "162741354")
no2 = Alphasense_Sensors("NO2-B43F", "202742056")
so2 = Alphasense_Sensors("SO2-B4", "164240348")
ox = Alphasense_Sensors("OX-B431", "204240461")

# to mV
we = df[preffix[0] + labels[0]]*1000
ae = df[preffix[0] + labels[1]]*1000
# temp = df[preffix[0] + 'temp']


ppb = ((we - co.electronic_we) - (ae - co.electronic_ae))/co.sensitivity

df[preffix[0] + 'co'] = ppb / 1000


#%%

print(df.describe())
#%%
Yco = df[label_ref]

Xco = df.loc[Yco.index][[preffix[0] + 'co', preffix[0] + 'co_we',preffix[0] + 'co_ae', preffix[0] + 'temp', 'pin_umid']]

X_train, X_valid, y_train, y_valid = train_test_split(Xco, Yco, train_size=0.6)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

kfold = RepeatedKFold(n_splits = 5, n_repeats = 1)
# kfold = StratifiedKFold(n_splits = 5)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

param_grid = {"randomforestregressor__n_estimators": np.array([32, 64, 124]),
              "randomforestregressor__max_depth": [None, 32],
              "randomforestregressor__bootstrap" : [False, True],
              'randomforestregressor__max_features': ["sqrt", None],
              'randomforestregressor__criterion': ['squared_error' ]}# 'absolute_error', 'friedman_mse']}

regressor = make_pipeline(RandomForestRegressor())


gs = GridSearchCV(regressor, param_grid=param_grid, n_jobs=-1, verbose = 3,\
                  return_train_score=True, cv = kfold, error_score = 'raise')

    
res = gs.fit(X_train,y_train)

# %% Resultado da otimização
print(train_data := pd.DataFrame(res.cv_results_))

with open('tabela_treino.tex', 'w') as f:
    f.write(train_data.style.to_latex())
    
    
var = 'squared_error'
var2 = 'sqrt'
# mse = train_data.query("param_randomforestregressor__criterion == @var and param_randomforestregressor__max_features == @var2")
mse_df = train_data.query("param_randomforestregressor__criterion == @var")

with open('tabela_treino_mse.tex', 'w') as f:
    f.write(mse_df.style.to_latex())
    
mse_df = mse_df.sort_values('param_randomforestregressor__n_estimators', axis = 0)

# Plot the responses for different events and regions
plt.figure()
sns.lineplot(x="param_randomforestregressor__n_estimators", y="mean_test_score",
             #hue="param_randomforestregressor__max_features", # style="event",
             data=train_data)
plt.show()

#%%

# print("Linear Regression Model")
# print("Train Score: ", linReg.score(X_train, y_train))
# print("Test Score: ", linReg.score(X_test, y_test))
# print("Validation Score: ", r2_score(y_valid, linReg.predict(X_valid)))
# print("RMSE Score: ", 100*rmse(y_train, linReg.predict(X_train)))

# print(linReg.coef_)

# sns.regplot(x = y_valid, y = linReg.predict(X_valid))
# sns.regplot(x = y_test, y = linReg.predict(X_test))
# plt.gca().axline((0,0), slope=1)
# plt.show()


#%%

print("Random Forest Model")
print("Train Score: ", gs.score(X_train, y_train))
print("Test Score: ", gs.score(X_test, y_test))
print("Validation Score: ", r2_score(y_valid, gs.predict(X_valid)))
# print("RMSE Score: ", 100*rmse(y_train, gs.predict(X_train)))

plt.figure()
sns.regplot(x = y_valid, y = gs.predict(X_valid))
sns.regplot(x = y_test, y = gs.predict(X_test))
plt.gca().axline((0,0), slope=1)

plt.show()

# #%% Antes de tudo

# e1_rf = {'co' : pd.DataFrame(data=gs.predict(Xco), index=Xco.index)}
# # e1 = {'co' : df['e2sp_co']}
# e2_ref = {'co' : df['iag_co']}

# plot_data_by_time_and_regr_plot(e1, e2, labels = ['co'], latex_labels = 'co')

# #%%
# ## ['2023-03-18 10:00:00':'2023-03-22 10:00:00'].

# e1_rf = {'co' : pd.DataFrame(data=linReg.predict(Xco), index=Xco.index)}
# # e1 = {'co' : df['e2sp_co']}
# e2_ref = {'co' : df['iag_co']}

# plot_data_by_time_and_regr_plot(e1_rf, e2_ref, labels = ['co'], latex_labels = 'co')
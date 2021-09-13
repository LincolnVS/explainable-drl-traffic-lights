# decision tree for multioutput regression
from scipy.sparse import data
from sklearn.datasets import make_regression
from xgboost import XGBRegressor,XGBClassifier, DMatrix
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np 
from copy import deepcopy
import pandas as pd
# import warnings filter
from warnings import simplefilter
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


import ast

def convert_list_string_to_list_float(strings):
    # Converting string to list
    values = []
    for s in strings:
        value = []
        for x in s.split( ):
            try:
                value.append(float(x.strip(' []')))
            except:
                pass
        values.append(value)
    return values

# ignore all future warnings
simplefilter(action='ignore')

class XGBRegressor(XGBRegressor):
    def partial_fit(self, X, y, *params):
        super().fit(X, y,*params, xgb_model=super().get_booster())

def flatten(t):
    return [item for sublist in t for item in sublist]

def get_dataset(path):
    dataset = pd.read_csv(path,';')

    rewards = dataset['reward']
    obs = convert_list_string_to_list_float(dataset["actual_state"])
    next_obs = convert_list_string_to_list_float(dataset["next_state"])
    actions = dataset['action']

    targets_f = np.zeros((len(obs),26))
                
    for i, action in enumerate(actions):
        targets_f[i][action] = rewards[i]

    X, y = np.array(obs),targets_f
    return X,y

def split_test(X,y):
    X,Xt,y,yt = ttsplit(X, y, test_size=0.33, random_state=42)

    X_ = []
    y_ = []
    for _,y_index in KFold(n_splits=4).split(X):
        X_.append(X[y_index])
        y_.append(y[y_index])

    # define model
    model = MultiOutputRegressor(XGBRegressor())
    model2 = MultiOutputRegressor(XGBRegressor())

    print("Test 1")
    model.fit(X, y)
    # make a prediction
    print("\tFit all X and y:",mean_squared_error(yt,model.predict(Xt)))

    model2.fit(X_[0], y_[0] )
    model3 = deepcopy(model2)
    # make a prediction
    print("\tFit only X_[0] and y_[0]:",mean_squared_error(yt,model2.predict(Xt)))

    model3.partial_fit(X_[1], y_[1] )
    model4 = deepcopy(model3)
    # make a prediction
    print("\tFit partial X_[1] and y_[1]:",mean_squared_error(yt,model3.predict(Xt)))

    model4.partial_fit(X_[2], y_[2] )
    model5 = deepcopy(model4)
    # make a prediction
    print("\tFit partial X_[2] and y_[2]:",mean_squared_error(yt,model4.predict(Xt)))

    model5.partial_fit(X_[3], y_[3] )
    # make a prediction
    print("\tFit partial X_[3] and y_[3]:",mean_squared_error(yt,model5.predict(Xt)))

    print("Test 2")

    # define model
    model = MultiOutputRegressor(XGBRegressor())
    model2 = MultiOutputRegressor(XGBRegressor())

    # fit all the model
    model.fit(X, y)
    # make a prediction
    print("\tFit all X and y:",mean_squared_error(yt,model.predict(Xt)))

    model2.fit(flatten(X_[0:1]), flatten(y_[0:1]) )
    model3 = deepcopy(model2)
    # make a prediction
    print("\tFit only X_[0:1] and y_[0:1]:",mean_squared_error(yt,model2.predict(Xt)))

    model3.partial_fit(flatten(X_[0:2]),flatten(y_[0:2]))
    model4 = deepcopy(model3)
    # make a prediction
    print("\tFit partial X_[0:2] and y_[0:2]:",mean_squared_error(yt,model3.predict(Xt)))

    model4.partial_fit(flatten(X_[1:3]), flatten(y_[1:3]) )
    model5 = deepcopy(model4)
    # make a prediction
    print("\tFit partial X_[1:3] and y_[1:3]:",mean_squared_error(yt,model4.predict(Xt)))

    model5.partial_fit(flatten(X_[2:4]), flatten(y_[2:4] ))
    # make a prediction
    print("\tFit partial X_[2:4] and y_[2:4]:",mean_squared_error(yt,model5.predict(Xt)))


def objective(space):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    X,y = get_dataset("log/my_sdqn/20210820-161915_0.csv")
    X,Xt,y,yt = ttsplit(X, y, test_size=0.33, random_state=42)


    clf=MultiOutputRegressor(XGBRegressor(
                    n_estimators =180, max_depth = 18, gamma = 1.0084573146866125,
                    reg_alpha = 43.0,min_child_weight=4.0))

    
    clf.fit(X, y,verbose=False)
    

    pred = clf.predict(Xt)
    accuracy = mean_squared_error(yt, pred)
    print ("SCORE:", accuracy)
    return {'loss': accuracy, 'status': STATUS_OK }

def hypertune_parameters():
    # import packages for hyperparameters tuning


    space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 100,
                            trials = trials)

    print(best_hyperparams)
    pass


X,y = get_dataset("agent/configs_xqn/buffer.csv")

#split_test(X,y)

hypertune_parameters()


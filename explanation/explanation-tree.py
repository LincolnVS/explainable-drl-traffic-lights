from numpy.lib.npyio import save
from scipy.sparse import data
from sklearn.datasets import make_regression
from xgboost import XGBRegressor, plot_tree
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np 
from copy import deepcopy
import pandas as pd
# import warnings filter
from warnings import simplefilter
from matplotlib import pyplot as plt
import pickle
from pathlib import Path

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

    targets_f = np.zeros((len(obs),8))
                
    for i, action in enumerate(actions):
        targets_f[i][action] = rewards[i]

    X, y = np.array(obs),targets_f
    return X[:36000],y[:36000]


space={'colsample_bytree': 0.8901461136447317, 'gamma': 1.0166204891662554, 'max_depth': 50.0, 'min_child_weight': 20.0, 'n_estimators': 170.0, 'reg_alpha': 44.0, 'reg_lambda': 0.26795775921100234}

clf=MultiOutputRegressor(XGBRegressor(
                    n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma =  space['gamma'],
                    reg_alpha = space['reg_alpha'],reg_lambda = space['reg_lambda'],min_child_weight=space['min_child_weight']))

def load_model():
    return pickle.load(open(f"../model/xqn/xqn_agent_intersection_1_1.pickle", "rb"))

def save_model(model):
    pickle.dump(model, file = open(f"../model/xqn/xqn_agent_intersection_1_1.pickle", "wb"))
    


#X,y = get_dataset("../agent/configs_xqn/buffer.csv")
#clf.fit(X, y,verbose=False)

#save_model(clf)

model = load_model()


Path(f"tree/").mkdir(parents=True, exist_ok=True)

for idx,a in enumerate(model.estimators_):
    plot_tree(a)
    ax = plt.gca()
    fig = plt.gcf()
    #fig.set_size_inches(16,8)
    plt.savefig(f"tree/plot_tree_{idx}.pdf",dpi=1000,bbox_inches = 'tight')
    plt.close()


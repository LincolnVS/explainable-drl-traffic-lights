from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
import os
from agent.SamplerAlgorithms import ProportionalSampler2 as ProportionalSampler
import pickle
from numpy import loadtxt
from xgboost import XGBRegressor,XGBClassifier, DMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from copy import deepcopy


import os
from agent.SamplerAlgorithms import ProportionalSampler2 as ProportionalSampler
import time

import pandas as pd

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')

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
    return X,y


# class XGBRegressor(XGBRegressor):
#     def partial_fit(self, X, y):
#         super().fit(X, y, xgb_model=super().get_booster())

def current_milli_time():
    return round(time.time() * 1000)


class XQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid, parameters):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid
        self.parameters = parameters
        self.ob_length = ob_generator.ob_length


        self.gamma = self.parameters["gamma"]  # discount rate
        self.epsilon = self.parameters["epsilon_initial"]  # exploration rate
        self.epsilon_min = self.parameters["epsilon_min"]
        self.epsilon_decay = self.parameters["epsilon_decay"]

        self.batch_size = self.parameters['batch_size']
        self.memory = deque(maxlen=self.parameters["buffer_size"])
        
        self.sampler_algorithm = ProportionalSampler(self.parameters["buffer_size"],self.parameters['batch_size'])

        self.learning_start = self.parameters["learning_start"]
        self.update_model_freq = self.parameters["update_model_freq"]
        self.update_target_model_freq = self.parameters["update_target_model_freq"]
        self.epochs_replay = self.parameters["epochs_replay"]
        self.epochs_initial_replay = self.parameters["epochs_initial_replay"]

        self.model = self._build_model()
        self.flag_ini_train = True

        self.ds_path = parameters['dataset_path'] 


    def get_action(self, ob):
        if np.random.rand() <= self.epsilon or self.flag_ini_train:
            return self.action_space.sample()
        ob = self._reshape_ob(ob)
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])

    def sample(self):
        return self.action_space.sample()

    def _build_model(self):
        
        model = MultiOutputRegressor(XGBRegressor(n_estimators =180, max_depth = 18, min_child_weight=4))

        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def _reshape_ob_array(self, ob_array):
        narray = [np.reshape(ob, (1, -1)) for ob in ob_array]
        return narray
    def update_target_network(self):
        #if self.total_decision > self.learning_start and not(self.total_decision%self.update_target_model_freq):
            #self.target_model = deepcopy(self.model)
        pass
    
    def remember(self, ob,  action, reward, next_ob):
        #PASSAR LISTA PARA O DEQUE (não array) SE NÃO DA MERDA
        self.memory.append([ob, action, reward, next_ob])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = self.sampler_algorithm.get_sample(self.memory,len(self.memory))
        #print(len(minibatch))
        _obs, actions, rewards, _next_obs = [np.stack(x) for x in np.array(minibatch).T]
        obs =  np.array(_obs)
        next_obs =  np.array(_next_obs)

        if self.flag_ini_train:
            target = rewards
            target_f = np.zeros((len(obs),self.action_space.n))   
            self.flag_ini_train = False
        else:
            target = rewards + self.gamma * np.amax(self.model.predict(next_obs), axis=1)
            target_f = self.model.predict(obs)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        self.model.fit(obs, target_f)
        self.decay_epsilon()

    def load_model(self, dir="model/xqn"):
        name = "xqn_agent_{}.pickle".format(self.iid)
        model_name = os.path.join(dir, name)
        #self.model.load_weights(model_name)
        self.model = pickle.load(open(f"{model_name}", "rb"))

    def save_model(self, dir="model/xqn"):
        name = "xqn_agent_{}.pickle".format(self.iid)
        model_name = os.path.join(dir, name)
        
        pickle.dump(self.model, file = open(f"{model_name}", "wb"))
    
    def decay_epsilon(self):
        self.epsilon = np.max([self.epsilon*self.epsilon_decay,self.epsilon_min])

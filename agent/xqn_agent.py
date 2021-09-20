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

    targets_f = np.zeros((len(obs),26))
                
    for i, action in enumerate(actions):
        targets_f[i][action] = rewards[i]

    X, y = np.array(obs),targets_f
    return X,y


class XGBRegressor(XGBRegressor):
    def partial_fit(self, X, y):
        super().fit(X, y, xgb_model=super().get_booster())

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
        self.first_replay = True

        self.ds_path = parameters['dataset_path'] 


    def get_action(self, ob):
        if np.random.rand() <= self.epsilon:
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
        if self.total_decision > self.learning_start and not(self.total_decision%self.update_target_model_freq):
            #self.target_model = deepcopy(self.model)
            pass
    
    def remember(self, ob,  action, reward, next_ob):
        #PASSAR LISTA PARA O DEQUE (não array) SE NÃO DA MERDA
        self.memory.append([ob, action, reward, next_ob])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = self.sampler_algorithm.get_sample(self.memory)
        
        _obs, actions, rewards, _next_obs = [np.stack(x) for x in np.array(minibatch).T]
        obs,phase = [np.stack(x) for x in np.array(_obs).T]
        obs =  np.array(obs)
        next_obs,next_phase = [np.stack(x) for x in np.array(_next_obs).T]
        next_obs =  np.array(next_obs)
        if self.flag_treino_inicial:
            #monta arvore
            print("--Montar Arvore no Primeiro Replay--")
            targets_f = np.zeros((len(obs),self.action_space.n))
                        
            for i, action in enumerate(actions):
                targets_f[i][action] = rewards[i]
            #target = DMatrix(targets_f,label=obs)
            self.model.fit(obs, targets_f,verbose=1)
            self.target_model = deepcopy(self.model)
            
            self.flag_treino_inicial=False 
            return
        q2 = self.target_model.predict(next_obs)
        delta = rewards + self.gamma * np.max(q2, axis=1)
        targets = self.model.predict(obs)
        
        n_obs = []
        dict_obs = {}

         for i,action in enumerate(actions):
            if obs[i].tolist() in n_obs:
                key = str(obs[i].tolist())
                valores = dict_obs[key]
                dict_obs[key].append(i)
                last_idx = valores[-1]
                del valores[-1]
                targets[last_idx][action] = delta[i]
                targets[i] = targets[last_idx]
                for v in valores:
                    targets[v] = targets[i]
            else:
                key = str(obs[i].tolist())
                n_obs.append(obs[i].tolist())
                dict_obs[key] = [i]
                targets[i][action] = delta[i] 
        
        #print(obs, targets)
        self.model.fit(obs, targets)
        #print(f"loss: {np.mean(history.history['loss'])}") if self.flag_treino_inicial else None

    def load_model(self, dir="model/dqn"):
        name = "xqn_agent_{}.pickle".format(self.iid)
        model_name = os.path.join(dir, name)
        #self.model.load_weights(model_name)
        self.model = pickle.load(open(f"{model_name}", "rb"))

    def save_model(self, dir="model/dqn"):
        name = "xqn_agent_{}.pickle".format(self.iid)
        model_name = os.path.join(dir, name)
        
        pickle.dump(self.model, file = open(f"{model_name}", "wb"))
    
    
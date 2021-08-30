
from . import RLAgent
import random
import numpy as np
from collections import deque

from numpy import loadtxt
from xgboost import XGBRegressor,XGBClassifier, DMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from copy import deepcopy


import os
from agent.SamplerAlgorithms import ProportionalSampler2 as ProportionalSampler
import time

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')

class XGBRegressor(XGBRegressor):
    def partial_fit(self, X, y):
        super().fit(X, y, xgb_model=super().get_booster())

def current_milli_time():
    return round(time.time() * 1000)

class XQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, i, world,log_path,info_file):
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator
        super().__init__(action_space, ob_generator, reward_generator)
        
        self.action_space = action_space

        self.log_path = log_path  

        self.I = i
        self.iid = i.id

        self.ob_shape = ob_generator.ob_shape

        self.world = world
        self.world.subscribe("pressure")
        self.world.subscribe("car_count")
        self.world.subscribe("lane_count")

        self.batch_size = info_file['batch_size']
        self.buffer_size = self.batch_size*info_file['batch_size']
        self.memory = deque(maxlen=self.buffer_size)
        self.learning_start = info_file['learning_start']
        self.update_model_freq = info_file['update_model_freq'] #cada 25 ciclos 
        self.update_target_model_freq = info_file['update_target_model_freq']

        self.gamma = info_file['gamma'] # discount rate
        self.epsilon_start = info_file['epsilon_start'] #1 # exploration rate
        self.epsilon = self.epsilon_start 
        self.epsilon_min = info_file['epsilon_min'] 
        self.epsilon_decay = info_file['epsilon_decay'] #.1/175 #0.975
        self.decay_type = info_file['decay_type']

        self.learning_rate_model = info_file['learning_rate_model']
        self.epochs_first_replay =info_file['epochs_first_replay']
        self.epochs_replay = info_file['epochs_replay']
        
        self.activation=info_file['activation']

        self.actual_phase = -1
        self.last_phase = -1
        self.total_decision = 0
        self.n_phases = len(self.I.phases)
        self.start_phase = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.sampler_algorithm = ProportionalSampler(self.buffer_size,self.batch_size)
        self.flag_treino_inicial = True

        self.current_reward = []
        self.episode_action_time = 0
        self.action_time = 0

        self.random_list_values = [0,5,10,15,20,25]
        self.random_coefficient = (self.learning_start+1)/len(self.random_list_values)

    def get_phase(self):
        return self.actual_phase

    def next_phase(self,phase):
        if phase < self.start_phase+self.n_phases-1:
            return phase + 1
        else:
            return self.start_phase

    def change_phase(self):
        if self.actual_phase == -1:
            self.actual_phase = self.start_phase
            self.last_phase = self.start_phase + self.n_phases - 1
            return
        self.last_phase = self.actual_phase
        
        if self.actual_phase < self.start_phase+self.n_phases-1:
            self.actual_phase += 1
        else:
            self.actual_phase = self.start_phase

    def get_reward(self):
        pressures = self.world.get_info("pressure")[self.iid]*0.005
        return -1*pressures

    def get_action(self, ob):
        self.total_decision += 1

        if self.total_decision < self.learning_start:

            value = 5*round(self.total_decision/200)
            if value > self.action_space.n:
                value = random.choice([0,5,10,15,20,25])
            value = np.floor(self.total_decision/self.random_coefficient).astype(int)
            return self.random_list_values[value]
            #return self.sample()

        else:
            if np.random.rand() <= self.epsilon:
                return self.sample()
            if self.flag_treino_inicial:
                return self.sample()

            ob = self._reshape_ob(ob)
            
            act_values = self.model.predict(ob)[0]

            #print(act_values)
            return np.argmax(act_values)

    def reset_episode_infos(self):
        self.actual_phase = -1
        self.last_phase = -1
        self.current_reward = []
        self.episode_action_time = 0
        self.action_time = 0

    def sample(self):
        return np.random.randint(0, self.action_space.n)

    def _build_model(self):
        
        model = MultiOutputRegressor(XGBRegressor(n_estimators =180, max_depth = 18, min_child_weight=4))

        return model

    def _reshape_ob(self, ob):
        #return np.reshape(ob, (1, -1))
        return np.array(ob)[None]

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

        if self.total_decision < self.learning_start:
            return 
        if self.total_decision % self.update_model_freq: 
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
        self.model.partial_fit(obs, targets)
        #print(f"loss: {np.mean(history.history['loss'])}") if self.flag_treino_inicial else None


    def load_model(self, dir="model/dqn"):
        name = "xqn_agent_{}.json".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.load_model(model_name)

    def save_model(self, dir="model/dqn"):
        name = "xqn_agent_{}.json".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.save_model(model_name)

    def decay_epsilon(self):
        if self.decay_type == "linear":
            self.epsilon = np.max([self.epsilon-self.epsilon_decay,self.epsilon_min])
        else:
            self.epsilon = np.max([self.epsilon*self.epsilon_decay,self.epsilon_min])


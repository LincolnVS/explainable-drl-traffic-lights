
from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.utils import plot_model
from keras.models import Sequential,Model
from keras.layers import Dense,Input,concatenate
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
from tensorflow.keras import initializers
from keras.losses import Huber
import os
from agent.SamplerAlgorithms import ProportionalSampler2 as ProportionalSampler
import time

def current_milli_time():
    return round(time.time() * 1000)

class DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, i, world,log_path,info_file):
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator
        super().__init__(action_space, ob_generator, reward_generator)
        
        self.log_path = log_path     
        self.state_with_phase = False

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

        last = action_space.n
        self.random_list_values = list(range(0,last,int(last/6)))
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

    def get_pressures_reward(self):
        #print(self.world.get_info("pressure"))
        pressure = self.world.get_info("pressure")[self.iid]
        return pressure

    def real_reward(self,first_obs,current_obs):
        reward = np.subtract(first_obs,current_obs)
        weights = [1.5,1.5,1]#[2,1.5,1]
        reward =  np.multiply(reward,weights)
        return np.round(np.sum(reward),3)*0.01
        #return -self.get_pressures_reward()/100

    def get_reward(self,last_pressure=0):
        pressures = self.get_pressures_reward()*0.005

        #print(wait)
        return -1*pressures

    def get_action(self, ob):
        self.total_decision += 1

        if self.total_decision < self.learning_start:

            value = 5*round(self.total_decision/200)
            if value > self.action_space.n:
                value = random.choice(self.random_list_values)
            value = np.floor(self.total_decision/self.random_coefficient).astype(int)
            return self.random_list_values[value]
            #return self.sample()

        else:
            if np.random.rand() <= self.epsilon:
                return self.sample()

            ob = self._reshape_ob(ob)
            
            if self.state_with_phase:
                phase = self._reshape_ob(self.actual_phase/7)
                act_values = self.model.predict([ob,phase])
            else:
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
        # Neural Net for Deep-Q learning Model
        
        model = Sequential()
        model.add(Dense(35, activation='relu', name='fc_input', input_shape=self.ob_shape))

        #model.add(Dense(512, activation="relu", name='fc_h1',kernel_initializer = initializer))
        #model.add(Dense(512, activation="relu", name='fc_h2',kernel_initializer = initializer))
        model.add(Dense(35, activation="relu", name='fc_h3') )      
        model.add(Dense(self.action_space.n, activation=self.activation,name='fc_output') )
        model.summary()

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        plot_model(model, f"{self.log_path}_model.png", show_shapes=True)

        return model

    def _reshape_ob(self, ob):
        #return np.reshape(ob, (1, -1))
        return np.array(ob)[None]

    def _reshape_ob_array(self, ob_array):
        narray = [np.reshape(ob, (1, -1)) for ob in ob_array]
        return narray

    def update_target_network(self):
        if self.total_decision > self.learning_start and not(self.total_decision%self.update_target_model_freq):
            weights = self.model.get_weights()
            self.target_model.set_weights(weights)

    def remember(self, ob,  action, reward, next_ob):
        #PASSAR LISTA PARA O DEQUE (não array) SE NÃO DA MERDA
        self.memory.append([ob, action, reward, next_ob])

    def replay(self):

        if self.total_decision < self.learning_start:
            return None 
        if self.total_decision % self.update_model_freq: 
            return None 

        minibatch = self.sampler_algorithm.get_sample(self.memory)
        
        _obs, actions, rewards, _next_obs = [np.stack(x) for x in np.array(minibatch).T]

        obs,phase = [np.stack(x) for x in np.array(_obs).T]
        obs =  np.array(obs)
        next_obs,next_phase = [np.stack(x) for x in np.array(_next_obs).T]
        next_obs =  np.array(next_obs)
        #t_next_obs =  self._reshape_ob_array(next_obs)

        q2 = self.target_model.predict(next_obs)
        
        if self.flag_treino_inicial:
            delta = rewards
            targets = np.zeros_like(self.model.predict(obs))
        else:
            delta = rewards + self.gamma * np.max(q2, axis=1)
            targets = self.model.predict(obs)

        #print('\n--------------------')
        #print(obs[0],actions[0],rewards[0],next_obs[0],delta[0],targets[0][actions[0]])
        
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
        
        epocas = self.epochs_first_replay if self.flag_treino_inicial else self.epochs_replay
        history = self.model.fit(obs, targets, epochs=epocas, verbose=0)
        print(f"loss: {np.mean(history.history['loss'])}") if self.flag_treino_inicial else None

        self.flag_treino_inicial = False

    def load_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

    def decay_epsilon(self):
        if self.decay_type == "linear":
            self.epsilon = np.max([self.epsilon-self.epsilon_decay,self.epsilon_min])
        else:
            self.epsilon = np.max([self.epsilon*self.epsilon_decay,self.epsilon_min])

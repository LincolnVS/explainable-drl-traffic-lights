from world import Intersection
from . import RLAgent
import random
import numpy as np
from collections import deque
import os
from itertools import combinations, product
import pickle

class FourierBasis:

    def __init__(self, state_dim, action_dim, order, max_non_zero=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.order = order
        self.max_non_zero = min(max_non_zero, state_dim)
        self.coeff = self._build_coefficients()
    
    def get_learning_rates(self, alpha):
        #lrs = np.linalg.norm(np.vstack((self.coeff, self.coeff)), axis=1)  # aqui sin
        lrs = np.linalg.norm(self.coeff, axis=1) 
        lrs[lrs==0.] = 1.
        lrs = alpha/lrs
        return lrs
    
    def _build_coefficients(self):
        coeff = np.array(np.zeros(self.state_dim), dtype=np.float32)  # Bias
        for i in range(1, self.max_non_zero + 1):
            for indices in combinations(range(self.state_dim), i):
                for c in product(range(1, self.order + 1), repeat=i):
                    coef = np.zeros(self.state_dim, dtype=np.float32)
                    coef[list(indices)] = list(c)
                    coeff = np.vstack((coeff, coef))
        return coeff
    
    def get_features(self, state):
        x = np.cos(np.dot(np.pi*self.coeff, state))
        return x

    def get_num_basis(self) -> int:
        #return len(self.coeff)*2  # aqui sin
        return len(self.coeff)

class TOSFB(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, intersection, world):
        super().__init__(action_space, ob_generator, reward_generator)

        self.intersection = intersection
        self.iid = intersection.id
        
        self.world = world
        self.world.subscribe("pressure")

        self.action_dim = action_space.n
        self.state_dim = ob_generator.ob_length
        self.learning_start = 0

        self.gamma = 0.9  # discount rate
        self.alpha = 0.0005
        self.lr = self.alpha
        self.lamb = 0 #0.9
        self.epsilon = 0.01
        self.epsilon_decay = 1 #0.9995 #0.9995
        self.min_epsilon = 0.01
        self.fourier_order = 9 #9
        self.max_non_zero_fourier = 2
        basis = 'fourier'

        if basis == 'fourier':
            self.basis = FourierBasis(self.state_dim, self.action_dim, self.fourier_order, max_non_zero=self.max_non_zero_fourier)
            self.lr = self.basis.get_learning_rates(self.alpha)
        self.num_basis = self.basis.get_num_basis()

        self.et = {a: np.zeros(self.num_basis, dtype=np.float32) for a in range(self.action_dim)}
        self.theta = {a: np.zeros(self.num_basis, dtype=np.float32) for a in range(self.action_dim)}

        self.use_buffer = True
        self.buffer = deque(maxlen=1000)

        self.q_old = None
        self.action = None
        self.td_error = 0

        #Phase Vars:
        self.n_phases = len(self.intersection.phases)
        self.start_phase = 0
        
        #Simulation Vars:
        self.action_time = 0 #Tempo escolhido pelo agente
        self.real_time = 0 #Tempo real (10+action_time*2)
        self.times_skiped = 0 #Quantidade de vezes que já estamos esperando
        self.obs = [] #Informação do estado
        self.reward = [] #Informação de recompesa
        self.phase = 0 #Fase atual

    def next_phase(self,phase):
        if phase < self.start_phase+self.n_phases-1:
            return phase + 1
        else:
            return self.start_phase

    def change_phase(self):        
        self.phase = self.next_phase(self.phase)

    def get_q_value(self, features, action):
        return np.dot(self.theta[action], features)
        
    def get_features(self, state):
        return self.basis.get_features(state)

    def reset_traces(self):
        self.q_old = None
        for a in range(self.action_dim):
            self.et[a].fill(0.0)

    def get_action(self, obs):
        features = self.get_features(obs)
        return self.act(features)

    def get_reward(self):
        reward = self.world.get_info("pressure")[self.iid]*0.005
        return -1*reward

    def act(self, features):
        if np.random.rand() < self.epsilon:
            #self.reset_traces()
            return self.action_space.sample()
        else:
            q_values = [self.get_q_value(features, a) for a in range(self.action_dim)]
            return q_values.index(max(q_values))

    def sample(self):
        return self.action_space.sample()

    def remember(self, state, action, reward, next_state, done=False):
        if self.use_buffer:
            self.buffer.append((state, action, reward, next_state))
            minibatch = [(state, action, reward, next_state)] + random.sample(self.buffer, min(31, len(self.buffer)))
        else:
            minibatch = [(state, action, reward, next_state)]
        for sample in minibatch:
            state, action, reward, next_state = sample
            phi = self.get_features(state)
            next_phi = self.get_features(next_state)
            q = self.get_q_value(phi, action)
            if not done:
                """ q_values = [self.get_q_value(next_phi, a) for a in range(self.action_dim)]
                next_q = max(q_values) """
                next_q = self.get_q_value(next_phi, self.act(next_phi))
            else:
                next_q = 0.0
            td_error = reward + self.gamma * next_q - q
            self.td_error = td_error
            if self.q_old is None:
                self.q_old = q

            for a in range(self.action_dim):
                if a == action:
                    self.et[a] = self.lamb*self.gamma*self.et[a] + (1 - self.lr*self.gamma*self.lamb*np.dot(self.et[a],phi))*phi
                    self.theta[a] += self.lr*(td_error + q - self.q_old)*self.et[a] - self.lr*(q - self.q_old)*phi
                else:
                    self.et[a] = self.lamb*self.gamma*self.et[a]
                    self.theta[a] += self.lr*(td_error + q - self.q_old)*self.et[a]
            
            self.q_old = next_q
            if done:
                self.reset_traces()
        
        self.epsilon = max(self.epsilon_decay*self.epsilon, self.min_epsilon)

    """ def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs = [np.stack(x) for x in np.array(minibatch).T]

        #target_values_predict = np.amax(self.target_model.predict(next_obs), axis=1)
        #print(rewards[0], target_values_predict[0])

        #target = rewards + self.gamma * target_values_predict
        #target_f = self.model.predict(obs)

        target = rewards + self.gamma * np.amax(self.target_model.predict(next_obs), axis=1)
        
        target_f = self.model.predict(obs)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        history = self.model.fit(obs, target_f, epochs=1, verbose=0)
        #print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay """

    def load_model(self, dir="model/tosfb"):
        with open(dir+'/model.pickle"', 'rb') as f:
            self = pickle.load(f)

    def save_model(self, dir="model/tosfb"):
        world = self.world
        ob = self.ob_generator
        rg = self.reward_generator
        intersection = self.intersection
        self.intersection = None
        self.world = None
        self.ob_generator = None
        self.reward_generator = None
        with open(dir+'/model.pickle', 'wb+') as f:
            pickle.dump(self, f)
        self.world = world
        self.ob_generator = ob
        self.reward_generator = rg
        self.intersection = intersection
    
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

class DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid, parameters,world):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid
        self.parameters = parameters
        self.ob_length = ob_generator.ob_length
        
        self.world = world
        self.world.subscribe("pressure")


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
        self.target_model = self._build_model()
        self.update_target_network()
        self.first_replay = True

    def get_action(self, ob):
        if np.random.rand() <= self.epsilon:

            return np.argmax(ob)
        ob = self._reshape_ob(ob)
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])

    #def get_reward(self):
    #    reward = self.world.get_info("pressure")[self.iid]*0.005
    #    return -1*reward

    def sample(self):
        return self.action_space.sample()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.parameters["hiden_nodes"], input_dim=self.ob_length, activation=self.parameters["hiden_activation"]))
        for _ in range(1,self.parameters["hiden_layers"]):
            model.add(Dense(self.parameters["hiden_nodes"], activation=self.parameters["hiden_activation"]))
        model.add(Dense(self.action_space.n, activation=self.parameters["output_activation"]))
        model.compile(
            loss= self.parameters["loss"],
            optimizer= self.parameters["optimizer"]
        )
        
        plot_model(model, f"{self.parameters['log_path']}/model.png", show_shapes=True)

        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        #minibatch = random.sample(self.memory, self.batch_size)
        minibatch = self.sampler_algorithm.get_sample(self.memory)
        obs, actions, rewards, next_obs = [np.stack(x) for x in np.array(minibatch).T]

        target = rewards + self.gamma * np.amax(self.target_model.predict(next_obs), axis=1)
        
        target_f = self.model.predict(obs)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        if self.first_replay:
            history = self.model.fit(obs, target_f, epochs=self.parameters['epochs_initial_replay'], verbose=0)
            self.first_replay=False
        else:
            history = self.model.fit(obs, target_f, epochs=self.parameters['epochs_replay'], verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.pickle".format(self.iid)
        model_name = os.path.join(dir, name)
        #self.model.load_weights(model_name)
        self.model = pickle.load(open(f"{model_name}", "rb"))

    def save_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.pickle".format(self.iid)
        model_name = os.path.join(dir, name)
        
        pickle.dump(self.model, file = open(f"{model_name}", "wb"))
    
    
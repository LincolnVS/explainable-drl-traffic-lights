from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.utils import plot_model
from keras.models import Sequential,Model
from keras.layers import Dense,Input,concatenate
from keras.optimizers import Adam, RMSprop, SGD
import os
from agent.SamplerAlgorithms import ProportionalSampler2 as ProportionalSampler
import time

def current_milli_time():
    return round(time.time() * 1000)

class DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, i, world):
        super().__init__(action_space, ob_generator, reward_generator)
        
        self.state_with_phase = False

        self.I = i
        self.iid = i.id

        self.ob_length = ob_generator.ob_length

        self.world = world
        self.world.subscribe("pressure")
        self.world.subscribe("car_count")
        self.world.subscribe("lane_count")

        self.batch_size = 2000
        self.memory = deque(maxlen=4000)
        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 1

        self.gamma = 0.99  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        
        self.actual_phase = -1
        self.last_phase = -1
        self.total_decision = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.sampler_algorithm = ProportionalSampler(len(self.memory),self.batch_size)
        self.flag_treino_inicial = True

        self.current_reward = []
        self.episode_action_time = 0
        self.action_time = 0

    def get_phase(self):
        return self.actual_phase

    def next_phase(self,phase):
        if phase < len(self.I.phases)-1:
            return phase + 1
        else:
            return 0

    def change_phase(self):
        if self.actual_phase == -1:
            self.actual_phase = 0
            self.last_phase = 7
            return
        self.last_phase = self.actual_phase
        if self.actual_phase < len(self.I.phases)-1:
            self.actual_phase += 1
        else:
            self.actual_phase = 0

    def get_pressures_reward(self):
        #print(self.world.get_info("pressure"))
        pressure = self.world.get_info("pressure")[self.iid]
        return pressure


    def get_reward(self,last_pressure=0):
        pressures = self.get_pressures_reward()/100

        #print(wait)
        return -1*pressures

    def get_action(self, ob):
        self.total_decision += 1

        if self.total_decision < self.learning_start:
            #return self.sample()
            return np.random.choice([self.sample(),0,5,10,15], 1, p=[0.2, 0.1, 0.3, 0.3, 0.1])[0]
            #return np.random.choice([0,5,10,15], 1)[0]
        else:
            if np.random.rand() <= self.epsilon:
                return self.sample()

            ob = self._reshape_ob(ob)
            if self.state_with_phase:
                phase = self._reshape_ob(self.actual_phase/7)
                act_values = self.model.predict([ob,phase])
            else:
                act_values = self.model.predict(ob)

            #print(act_values)
            #print([ob,phase])

            return np.argmax(act_values[0])
    def reset_episode_infos(self):
        self.actual_phase = -1
        self.last_phase = -1
        self.current_reward = []
        self.episode_action_time = 0
        self.action_time = 0

    def sample(self):
        return np.random.randint(0, 23)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        if self.state_with_phase:
            inputA = Input(shape= (self.ob_length,), name='state_input')
            inputB = Input(shape=(1,), name='phase_input')

            A = Model(inputs=inputA, outputs=inputA)

            B = Model(inputs=inputB, outputs=inputB)
            
            combined = concatenate([A.output,B.output])
            z = Dense(16, activation="relu", name='fc_h0d')(combined)
            z = Dense(256, activation="relu", name='fc_h1')(z)
            z = Dense(529, activation="relu", name='fc_h2')(z)
            #z = Dense(20, activation="relu", name='fc_h2')(z)
            #z = Dense(512, activation="relu", name='fc_h3')(z)
            z = Dense(self.action_space.n,activation="linear",name='fc_output')(z)
            model = Model(inputs=[A.input, B.input], outputs=z)

            model.compile(optimizer=RMSprop(lr=0.001),
                            loss='mse')
            model.summary()
            #model.compile(loss='mse', optimizer='adam')
            plot_model(model, "model.png", show_shapes=True)
        else:
            model = Sequential()
            model.add(Dense(20, activation='relu', name='fc_input', input_shape=(self.ob_length,)))
            model.add(Dense(self.action_space.n*2, activation="relu", name='fc_h1'))
            #model.add(Dense(20, activation="relu", name='fc_h2'))
            #model.add(Dense(512, activation="relu", name='fc_h3'))
            model.add(Dense(self.action_space.n,activation="tanh",name='fc_output'))
            model.compile(loss='mse', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
            model.summary()
            
            plot_model(model, "model.png", show_shapes=True)


        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        if self.total_decision > self.learning_start and not(self.total_decision%self.update_target_model_freq):
            weights = self.model.get_weights()
            self.target_model.set_weights(weights)

    def remember(self, ob,  action, reward, next_ob):
        self.memory.append((ob,action, reward, next_ob))

    def replay(self):
        #minibatch = random.sample(self.memory, self.batch_size)

        #print('Get Sample buffer')
        indexes = self.sampler_algorithm.get_sample(self.memory)

        minibatch = [self.memory[i] for i in indexes]

        _obs, actions, rewards, _next_obs = [np.stack(x) for x in np.array(minibatch).T]

        obs,phase = [np.stack(x) for x in np.array(_obs).T]
        next_obs,next_phase = [np.stack(x) for x in np.array(_next_obs).T]
 
        if self.state_with_phase:
            #print('Get next values')
            target = self.model.predict([obs,phase])
            q_next_states = self.target_model.predict([next_obs,next_phase])

            
            #print('Update values')
            # Update values according to Deep Q-Learnig algorithm
            for i, action in enumerate(actions):
                if self.flag_treino_inicial:
                    target[i][action] = rewards[i]
                else: # yj = rj + gamma * Q(s', a', theta)
                    target[i][action] = rewards[i] + self.gamma * np.max(q_next_states[i])

            #print('Fit Model')
            epocas = 1000 if self.flag_treino_inicial else 1
            history = self.model.fit([obs,phase], target, epochs=epocas, verbose=0)
        else:
            #print('Get next values')
            target = rewards + self.gamma * np.amax(self.target_model.predict(next_obs), axis=1)
            target_f = self.model.predict(obs)
            for i, action in enumerate(actions):
                target_f[i][action] = target[i]


            #print('Fit Model')
            epocas = 1000 if self.flag_treino_inicial else 1
            history = self.model.fit(obs, target_f, epochs=epocas, verbose=0)
            print(f"loss: {np.mean(history.history['loss'])}")

        self.flag_treino_inicial = False

        '''
        target = rewards + self.gamma * np.amax(self.target_model.predict([next_obs,next_phase]), axis=1)
        target_f = self.model.predict([obs,phase])
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        history = self.model.fit([obs,phase], target_f, epochs=1, verbose=0)
        '''
    def load_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

    def decay_epsilon(self):
        self.epsilon = np.max([self.epsilon*self.epsilon_decay,self.epsilon_min])

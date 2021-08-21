from . import BaseAgent
import math


class Fixedtime_Agent(BaseAgent):
    def __init__(self, action_space,options, I, world, ob_generator=None):
        super().__init__(action_space)
        self.I = I
        self.world = world
        self.ob_generator = ob_generator
        self.world.subscribe("pressure")

        self.action = 0

        self.phase_time = options['phase_time']

    def get_ob(self):
        if self.ob_generator is not None:
            return self.ob_generator.generate() 
        else:
            return None

    def get_pressures_reward(self):
        #print(self.world.get_info("pressure"))
        pressure = self.world.get_info("pressure")[self.I.id]
        return pressure
        
    def get_reward(self,last_pressure=0):
        pressures = self.get_pressures_reward()*0.005

        #print(wait)
        return -1*pressures

    def get_action(self, world):

        if self.I.current_phase_time >= self.phase_time:
            self.action = (self.action + 1) % self.action_space.n

        return self.action
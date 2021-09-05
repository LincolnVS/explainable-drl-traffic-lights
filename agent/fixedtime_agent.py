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

    def get_reward(self):
        pressure = self.world.get_info("pressure")[self.I.id]*0.005
        return -1*pressure

    def get_action(self, world):
        if self.I.current_phase_time >= self.phase_time and self.I.current_phase == self.action:
            self.action = (self.action + 1) % self.action_space.n

        return self.action
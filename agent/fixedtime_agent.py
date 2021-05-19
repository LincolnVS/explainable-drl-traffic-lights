from . import BaseAgent
import math


class Fixedtime_Agent(BaseAgent):
    def __init__(self, action_space,phase_time, iid):
        super().__init__(action_space)
        self.iid = iid
        self.last_action = 0
        self.last_action_time = 0

        self.phase_time = phase_time

    def get_ob(self):
        return 0

    def get_reward(self):
        return 0

    def get_action(self, world):
        current_time = world.eng.get_current_time()

        if current_time - self.last_action_time >= self.phase_time:
            self.last_action = (self.last_action + 1) % self.action_space.n
            self.last_action_time = current_time
        
        return self.last_action
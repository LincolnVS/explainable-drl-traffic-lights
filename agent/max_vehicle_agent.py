from . import BaseAgent
import numpy as np

class MaxVehicleAgent(BaseAgent):
    """
    Agent using Max-Pressure method to control traffic light
    """
    def __init__(self, action_space,options, I, world, ob_generator, rw_generator):
        super().__init__(action_space)
        self.I = I
        self.world = world
        self.world.subscribe("lane_count")
        self.world.subscribe("pressure")
        self.ob_generator = ob_generator
        self.reward_generator = rw_generator
        
        # the minimum duration of time of one phase
        self.t_min = options['delta_t']

    def get_ob(self):
        return self.ob_generator.generate()

    def get_action(self, ob):
        
        if self.I.current_phase_time < self.t_min:
            return self.I.current_phase

        return np.argmax(ob)    


    def get_reward(self):
        reward = self.reward_generator.generate()
        assert len(reward) == 1
        return reward[0]

from . import BaseAgent
from generator import LaneVehicleGenerator

class SOTLAgent(BaseAgent):
    """
    Agent using Self-organizing Traffic Light(SOTL) Control method to control traffic light
    """
    def __init__(self, action_space,options, I, world,ob_generator=None):
        super().__init__(action_space)
        self.I = I
        self.world = world
        self.world.subscribe("lane_waiting_count")
        self.world.subscribe("pressure")
        self.ob_generator = ob_generator

        # the minimum duration of time of one phase
        self.t_min = options['green_time']
        self.t_max = 60

        # some threshold to deal with phase requests
        self.min_green_vehicle = options['green_v']
        self.max_red_vehicle = options['red_v']

    def get_ob(self):
        if self.ob_generator is not None:
            return self.ob_generator.generate() 
        else:
            return None

    def get_action(self, ob):
        lane_waiting_count = self.world.get_info("lane_waiting_count")

        action = self.I.current_phase
        if self.I.current_phase_time >= self.t_min:
            num_green_vehicles = sum([lane_waiting_count[lane] for lane in self.I.phase_available_startlanes[self.I.current_phase]])
            num_red_vehicles = sum([lane_waiting_count[lane] for lane in self.I.startlanes])
            num_red_vehicles -= num_green_vehicles
            
            if num_red_vehicles > self.max_red_vehicle and num_green_vehicles <= self.min_green_vehicle:
                action = (action + 1) % self.action_space.n
            elif self.I.current_phase_time >= self.t_max:
                action = (action + 1) % self.action_space.n
        return action

    def get_reward(self):
        pressure = self.world.get_info("pressure")[self.I.id]*0.005
        return -1*pressure

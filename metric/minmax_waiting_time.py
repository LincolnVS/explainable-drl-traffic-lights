from . import BaseMetric
import numpy as np

class MaxWaitingTimeMetric(BaseMetric):
    """
    Calculate the max travel time of all vehicles.
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """
    def __init__(self, world):
        self.world = world
        self.world.subscribe(["waiting_time_count"])
        self.name = "Max Waiting Time Metric"
        self.reset()

    def update(self, done=False):
        w_time = self.world.get_info("waiting_time_count")
        if not w_time:
            return 0
        
        max_w_time = max(w_time.values())
        self.max_wtime = max([self.max_wtime,max_w_time])
        return self.eval()

    def eval(self):
        return self.max_wtime

    def reset(self):
        self.max_wtime = 0

class MinWaitingTimeMetric(BaseMetric):
    """
    Calculate the min travel time of all vehicles (not zero).
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """
    def __init__(self, world):
        self.world = world
        self.world.subscribe(["waiting_time_count"])
        self.name = "Min Waiting Time Metric"
        self.reset()

    def update(self, done=False):
        w_time = np.array(list(self.world.get_info("waiting_time_count").values()))
        w_time = w_time[(w_time > 0)]

        if len(w_time) == 0:
            return 0

        min_w_time = min(w_time)
        self.min_wtime = min([self.min_wtime,min_w_time])
        return self.eval()

    def eval(self):
        return self.min_wtime

    def reset(self):
        self.min_wtime = np.inf

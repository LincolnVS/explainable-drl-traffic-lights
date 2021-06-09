import numpy as np
from . import BaseGenerator

class StateOfThreeGenerator(BaseGenerator):
    """
    Generate State or Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    I : Intersection object
    fns : list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay" and "pressure"
    in_only : boolean, whether to compute incoming lanes only
    average : None or str
        None means no averaging
        "road" means take average of lanes on each road
        "all" means take average of all lanes
    negative : boolean, whether return negative values (mostly for Reward)
    """
    def __init__(self, world, I, fns, in_only=False, average=None, negative=False):
        self.world = world
        self.I = I

        # get lanes of intersections
        self.lanes = []
        if in_only:
            roads = I.in_roads
        else:
            roads = I.roads
        for road in roads:
            from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
            self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        # calculate result dimensions
        #size = sum(len(x) for x in self.lanes)
        size = 3

        self.ob_shape = np.array([0,0,0]).shape

        self.average = average
        self.negative = negative

    def generate(self):

        ret = {fn:self.world.get_info(fn) for fn in self.fns}

        ret = ret[self.fns[0]].get(self.I.id,[0,0,0])
        if self.negative:
            ret = ret * (-1)
        #print(ret)
        return ret

if __name__ == "__main__":
    from world import World
    world = World("examples/config.json", thread_num=1)
    laneVehicle = LaneVehicleGenerator(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())
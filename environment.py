import gym
import numpy as np
import cityflow

class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.

    Parameters
    ----------
    world: World object
    agents: list of agent, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """
    def __init__(self, world, agents, metric):
        self.world = world
        
        self.eng = self.world.eng
        self.n_agents = len(self.world.intersection_ids)
        self.n = self.n_agents

        assert len(agents) == self.n_agents

        self.agents = agents
        action_dims = [agent.action_space.n for agent in agents]
        self.action_space = gym.spaces.MultiDiscrete(action_dims)

        if isinstance(metric, list):
            self.metric = metric
        else:
            self.metric = [metric]

    def update_metric(self):
        if self.world.eng.get_current_time()%5 == 0:
            for metric in self.metric:
                metric.update(done=False)

    def reset_metric(self):
        for metric in self.metric:
            metric.reset()

    def step(self, actions):
        assert len(actions) == self.n_agents

        self.world.step(actions)
        self.update_metric()

        obs = [agent.get_ob() for agent in self.agents]
        rewards = [agent.get_reward() for agent in self.agents]
        num_vehicles = self.world.count_vehicles()

        if num_vehicles == 0 and self.world.eng.get_current_time() > 20:
            dones = [True] * self.n_agents
        else:
            dones = [False] * self.n_agents

        infos = {"count_vehicles":num_vehicles}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        self.reset_metric()
        obs = [agent.get_ob() for agent in self.agents]
        return obs

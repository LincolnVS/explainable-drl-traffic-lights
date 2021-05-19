import json
import gym
from environment import TSCEnv
from world import World
from agent import Fixedtime_Agent
from metric import TravelTimeMetric, ThroughputMetric, FuelMetric, TotalCostMetric
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--phase_time', type=int, choices=range(1,60) ,default = 20, help = 'time of the phase')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(Fixedtime_Agent(action_space,args.phase_time, i.id))

# create metric
metric = ThroughputMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
def test(met, met_name):
    obs = env.reset()
    env.update_metric(met)
    for i in range(args.steps):
        actions = []
        try:
            for agent in agents:
                actions.append(agent.get_action(world))
            obs, rewards, dones, info = env.step(actions)
            for ind_m in range(len(env.metric)):
                env.metric[ind_m].update(done=False)
            if i % int(args.steps/5) == 0:
                print(i, "/", args.steps)
        except:
            break

    for ind_m in range(len(met_name)):
        print("{} is {:.4f}".format(met_name[ind_m], env.metric[ind_m].update(done=True)))

metric = [TravelTimeMetric(world), ThroughputMetric(world), FuelMetric(world), TotalCostMetric(world)]
metric_name = ["Average Travel Time", "Average throughput", "Average fuel cost", "Average total cost"]
test(metric, metric_name)
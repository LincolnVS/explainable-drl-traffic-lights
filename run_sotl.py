import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import SOTLAgent
from metric import TravelTimeMetric
import argparse

from metric import TravelTimeMetric, ThroughputMetric, FuelMetric, TotalCostMetric


# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--green_time', type=int, default=20, help='')
parser.add_argument('--green_v', type=int, default=20, help='')
parser.add_argument('--red_v', type=int, default=30, help='')
args = parser.parse_args()

options = {
    'green_time':args.green_time,
    'green_v':args.green_v,
    'red_v':args.red_v
}
# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(SOTLAgent(action_space,options, i, world))

# create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), FuelMetric(world), TotalCostMetric(world)]
metric_name = ["Average Travel Time", "Average throughput", "Average fuel cost", "Average total cost"]

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
actions = []
dones=[False]
steps = 0
#while not all(dones):
    
for i in range(args.steps):
    actions = []
    for agent_id, agent in enumerate(agents):
        actions.append(agent.get_action(obs[agent_id]))
    obs, rewards, dones, info = env.step(actions)
    
    if steps% 200 == 0:
        print(steps,info["count_vehicles"])
    steps += 1
    for ind_m in range(len(env.metric)):
        env.metric[ind_m].update(done=False)


for ind_m in range(len(metric_name)):
    print("{} is {:.4f}".format(metric_name[ind_m], env.metric[ind_m].update(done=True)))

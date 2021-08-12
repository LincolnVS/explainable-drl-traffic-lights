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
parser.add_argument('--phase_time', type=int, choices=range(1,61), default = 20, help = 'time of the phase')
args = parser.parse_args()

#Agent options
options = {
    'phase_time':args.phase_time
}

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(Fixedtime_Agent(action_space,options, i, world))

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), FuelMetric(world), TotalCostMetric(world)]
metric_name = ["Average Travel Time", "Average throughput", "Average fuel cost", "Average total cost"]

#Create env
env = TSCEnv(world, agents, metric)

obs = env.reset()
actions = []
steps = 0

#Walk through the steps
while steps < args.steps:

    actions = []
    #Get the agents' actions
    for agent_id, agent in enumerate(agents):
        actions.append(agent.get_action(obs[agent_id]))

    #Run steps
    obs, rewards, dones, info = env.step(actions)
    steps += 1

    #Update Metrics
    for ind_m in range(len(env.metric)):
        env.metric[ind_m].update(done=False)

    #Check if it's over by flag "Done"
    if all(dones) == True:
        print(i)
        break

#Print all metrics
for ind_m in range(len(metric_name)):
    print("{} is {:.4f}".format(metric_name[ind_m], env.metric[ind_m].update(done=True)))

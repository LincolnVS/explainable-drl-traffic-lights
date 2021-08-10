import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import MaxPressureAgent
from metric import TravelTimeMetric, ThroughputMetric, FuelMetric, TotalCostMetric
import argparse
# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--delta_t', type=int, default=10, help='how often agent make decisions')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)
# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(MaxPressureAgent(
        action_space, i, world, 
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=True)
    ))

metric = [TravelTimeMetric(world), ThroughputMetric(world), FuelMetric(world), TotalCostMetric(world)]
metric_name = ["Average Travel Time", "Average throughput", "Average fuel cost", "Average total cost"]

# create metric
#metric = TravelTimeMetric(world)
# create env
env = TSCEnv(world, agents, metric)
# simulate
obs = env.reset()
actions = []
i = 0

while i < args.steps:
    actions = []
    
    for agent_id, agent in enumerate(agents):
        actions.append(agent.get_action(obs[agent_id]))

    for _ in range(args.delta_t):  
        obs, rewards, dones, info = env.step(actions)
        #print(world.intersections[0]._current_phase, end=",")
        #print(obs, actions)
        for ind_m in range(len(env.metric)):
            env.metric[ind_m].update(done=False)
        
        i += 1

    if all(dones) == True:
        print(i)
        break

for ind_m in range(len(metric_name)):
    print("{} is {:.4f}".format(metric_name[ind_m], env.metric[ind_m].update(done=True)))

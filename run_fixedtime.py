import gym
from environment import TSCEnv
from world import World
from agent import Fixedtime_Agent
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric

import utils as u
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

#start wandb
u.wand_init("tlc",str(args.phase_time),'fixed_time')

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(Fixedtime_Agent(action_space,options, i, world))

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world)]

#Create env
env = TSCEnv(world, agents, metric)

for e in range(200):
    obs = env.reset()
    actions = []
    steps = 0
    episodes_rewards = 0

    #Walk through the steps
    while steps < args.steps:

        actions = []
        #Get the agents' actions
        for agent_id, agent in enumerate(agents):
            actions.append(agent.get_action(obs[agent_id]))

        #Run steps
        obs, rewards, dones, info = env.step(actions)
        steps += 1
        episodes_rewards += rewards[0]

        #Check if it's over by flag "Done"
        if all(dones) == True:
            print(i)
            break

    eval_dict = {}
    eval_dict["epsilon"]=0
    eval_dict["episode"]=e
    eval_dict["steps"]=3600
    eval_dict["mean_episode_reward"]=episodes_rewards/3600

    for metric in env.metric:
        eval_dict[metric.name]=metric.eval()
        
    u.wand_log(eval_dict)

#Print all metrics
for metric in env.metric:
    print("{} is {:.4f}".format(metric.name, metric.eval()))

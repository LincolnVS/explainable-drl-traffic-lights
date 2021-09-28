import gym
from environment import TSCEnv
from world import World
from agent import MaxPressureAgent
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric, MaxWaitingTimeMetric

import utils as u
import argparse
#Parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--delta_t', type=int, default=20, help='how often agent make decisions')
args = parser.parse_args()

#Agent options
options = {
    'delta_t':args.delta_t
}

#Create world
world = World(args.config_file, thread_num=args.thread)

#Create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(MaxPressureAgent(action_space,options, i, world))

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world), MaxWaitingTimeMetric(world)]

#Create env
env = TSCEnv(world, agents, metric)

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

print(f"\n--MaxPressure Results--")
print(f"Steps: {steps}")
print(f"Episodes Rewards: {episodes_rewards/steps:.4f}")
# for metric in env.metric:
#     print(f"{metric.name}: {metric.eval():.4f}")

#start wandb
u.wand_init("TLC - Results C2",f"MaxPressure: {options['delta_t']}", "MaxPressure")

eval_dict = {}
eval_dict["epsilon"]=0
eval_dict["steps"]=steps
eval_dict["mean_episode_reward"]=episodes_rewards/steps
for metric in env.metric:
    eval_dict[metric.name]=metric.eval()
    print(f"{metric.name}: {metric.eval():.4f}")

for e in range(200):
    eval_dict["episode"]=e

    u.wand_log(eval_dict)
import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import MaxPressureAgent
from metric import TravelTimeMetric
import argparse
import os
import logging
from datetime import datetime

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=6000, help='number of steps')
parser.add_argument('--delta_t', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--log_dir', type=str, default="log/Maxpressure", help='directory in which logs should be saved')
args = parser.parse_args()



if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S') + ".log"))
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)


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

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
actions = []
dones=[False]
steps = 0

last_action = [-1] * len(agents)
actions = [0] * len(agents)
last_obs = obs

while not all(dones):
    for agent_id, agent in enumerate(agents):
        actions[agent_id] = agent.get_action(obs[agent_id])
        
        if actions[agent_id] != last_action[agent_id]:
            logger.info(f"OBS: {obs[agent_id]} - {last_obs[agent_id]}; Action: {actions[agent_id]} - {last_action[agent_id]}")
            last_obs[agent_id] = obs[agent_id]
            last_action[agent_id] = actions[agent_id]

    obs, rewards, dones, info = env.step(actions)


    if steps% 200 == 0:
        print(steps,info["count_vehicles"])
    #print(world.intersections[0]._current_phase, end=",")
    #print(obs, actions)
    #print(env.eng.get_average_travel_time())
    #print(obs)
    #print(rewards)
    # print(info["metric"])
    steps += 1

print("Final Travel Time is %.4f" % env.metric.update(done=True))
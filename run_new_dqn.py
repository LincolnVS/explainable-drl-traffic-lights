import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator, StateOfThreeGenerator,PressureRewardGenerator
from agent import DQNAgent
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric,MaxWaitingTimeMetric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
import utils as u
import ntpath

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
#parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
#parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=20, help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/dqn", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/dqn", help='directory in which logs should be saved')
parser.add_argument('--parameters', type=str, default="agent/configs_new_dqn4/8.json", help='path to the file with informations about the model')

args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

file_name_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
file_name = f"{args.log_dir}/{file_name_time}"

fh = logging.FileHandler(file_name+'.log')
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

#Config File
parameters = u.get_info_file(args.parameters)
episodes = parameters['episodes']
parameters['log_path'] = args.log_dir
action_interval = parameters['action_interval']

#start wandb
u.wand_init("TLC - Results C2",f"new_dqn: {ntpath.basename(args.parameters)[:-5]}", "new_dqn")

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []

#parameters['buffer_size'] = parameters['buffer_size']*len(world.intersections)
#parameters['batch_size'] = parameters['batch_size']*len(world.intersections)

for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(DQNAgent(
        action_space,
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None, scale=.025),
        PressureRewardGenerator(world, i, scale=0.005, negative=True),
        i.id,
        parameters,
        world
    ))
    if args.load_model:
        agents[-1].load_model(args.save_dir)

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world), MaxWaitingTimeMetric(world)]

# create env
env = TSCEnv(world, agents, metric)

# train dqn_agent
def train(args, env):
    total_decision_num = 0
    for e in range(episodes):
        
        last_obs = env.reset()
        if e % args.save_rate == args.save_rate - 1:
            env.eng.set_save_replay(True)
            env.eng.set_replay_file("replay_%s.txt" % e)
        else:
            env.eng.set_save_replay(False)
        episodes_rewards = [0 for i in agents]
        episodes_decision_num = 0
        i = 0

        while i < args.steps:

            if i % action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        actions.append(agents[0].get_action(last_obs[agent_id]))
                    else:
                        actions.append(agents[0].sample())

                rewards_list = []
                for _ in range(action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)

                for agent_id, agent in enumerate(agents):
                    #u.append_new_line(file_name+f"_{agent_id}",[[last_obs[agent_id],-1], actions[agent_id], rewards[agent_id], [obs[agent_id],-1],e,i])
                    agents[0].remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                    episodes_rewards[agent_id] += rewards[agent_id]                
                    episodes_decision_num += 1
                    
                total_decision_num += 1
                last_obs = obs

                #for agent_id, agent in enumerate(agents):
                if total_decision_num > agents[0].learning_start and total_decision_num % agents[0].update_model_freq == agents[0].update_model_freq - 1:
                    agents[0].replay()
                if total_decision_num > agents[0].learning_start and total_decision_num % agents[0].update_target_model_freq == agents[0].update_target_model_freq - 1:
                    agents[0].update_target_network()
                    
                        
            #if all(dones):
            #    break

        if e % args.save_rate == args.save_rate - 1:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            #for agent in agents:
            agents[0].save_model(args.save_dir)
            #    break

        eval_dict = {}

        logger.info(f"episode:{e}/{episodes-1}, steps:{i}")
        eval_dict["episode"]=e
        eval_dict["steps"]=i
            
        for agent_id, agent in enumerate(agents):
            logger.info("\tagent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))
        
        for metric in env.metric:
            logger.info(f"\t{metric.name}: {metric.eval()}")
            eval_dict[metric.name]=metric.eval()
   
        eval_dict["epsilon"]=agents[0].epsilon
        eval_dict["mean_episode_reward"]=episodes_rewards[0] / episodes_decision_num
        
        u.wand_log(eval_dict)
    
    #for agent in agents:
    agents[0].save_model(args.save_dir)
    #    break

if __name__ == '__main__':
    # simulate
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    train(args, env)
    #test()
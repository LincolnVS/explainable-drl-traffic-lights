import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator, StateOfThreeGenerator,PressureRewardGenerator
from agent.dqn_agent import DQNAgent
from agent.n_tosfb import TOSFB
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric,MaxWaitingTimeMetric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
import wandb


# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=20, help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/tosfb", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/tosfb", help='directory in which logs should be saved')
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
    # define tempo de amarelo
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(TOSFB(
        action_space,
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
        PressureRewardGenerator(world, i, scale=.005, negative=True),
        i,
        world
    ))
    if args.load_model:
        agents[-1].load_model(args.save_dir)


wandb.init(project='TLC - Results C2', 
            name='tosfb', 
            save_code=True,
            config={'lr': agents[-1].alpha,
                    'fourier_order': agents[-1].fourier_order,
                    'gamma': agents[-1].gamma,
                    'min_epsilon': agents[-1].min_epsilon,
                    'lambda': agents[-1].lamb,
                    'max_nonzero_fourier': agents[-1].max_non_zero_fourier,
                    'epsilon_decay': agents[-1].epsilon_decay},
            group='tosfb')

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world), MaxWaitingTimeMetric(world)]


# create env
env = TSCEnv(world, agents, metric)

# train dqn_agent
def train(args, env):
    total_decision_num = 0
    for e in range(args.episodes):
        for agent in agents:
            agent.reset_traces()

        obs = env.reset()
        last_obs = obs
        for agent_id, agent in enumerate(agents):
            last_obs[agent_id] = np.array(last_obs[agent_id], dtype=np.float32)*0.01

        if e % args.save_rate == args.save_rate - 1:
            env.eng.set_save_replay(True)
            env.eng.set_replay_file("replay_%s.txt" % e)
        else:
            env.eng.set_save_replay(False)
        
        #for agent_id, agent in enumerate(agents):
        #    agent.obs = obs[agent_id]

        episodes_rewards = [0 for i in agents]
        td_errors = [0 for i in agents]
        episodes_decision_num = 0
        i = 0

        while i < args.steps:
            if i % args.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    if total_decision_num > agent.learning_start:
                        actions.append(agents[0].get_action(last_obs[agent_id]))
                    else:
                        actions.append(agents[0].sample())

                rewards_list = []
                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    for agent_id, agent in enumerate(agents):
                        obs[agent_id] = np.array(obs[agent_id], dtype=np.float32)*0.01

                    i += 1
                    rewards_list.append(rewards)
                
                rewards = np.mean(rewards_list, axis=0)

                for agent_id, agent in enumerate(agents):
                    agents[0].remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                    episodes_rewards[agent_id] += rewards[agent_id]
                    episodes_decision_num += 1
                    total_decision_num += 1
                    td_errors[agent_id] += agent.td_error
                
                last_obs = obs

            #if all(dones):
            #    break

        if e % args.save_rate == args.save_rate - 1:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            #for agent in agents:
            agents[0].save_model(args.save_dir)


        print(f"episode:{e}/{args.episodes}, epi total decisions: {episodes_decision_num}")

        eval_dict = {}
        eval_dict["episode"]=e
        eval_dict["steps"]=i  
        for metric in env.metric:
            print(f"\t{metric.name}: {metric.eval()}")
            eval_dict[metric.name]=metric.eval()

        mean_reward = {}
        mean_td_error = {}
        for agent_id, agent in enumerate(agents):
            mean_reward[agent_id] = episodes_rewards[agent_id] / episodes_decision_num
            mean_td_error[agent_id] = td_errors[agent_id] / episodes_decision_num
            print(f"\tmean reward: {mean_reward[agent_id]}")
            print(f"\tmean td error: {mean_td_error[agent_id]}")

        eval_dict["epsilon"]=agent.epsilon
        eval_dict["mean_episode_reward"]=np.mean(list(mean_reward.values()))
        eval_dict['mean_td_error']= np.mean(list(mean_td_error.values()))

        wandb.log(eval_dict)

    wandb.run.finish()

if __name__ == '__main__':
    # simulate
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    train(args, env)
    #test()
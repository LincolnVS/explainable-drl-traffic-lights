import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator, StateOfThreeGenerator
from agent.my_dqn_agent import DQNAgent
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
import utils as u
# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=1, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=20, help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/my_sdqn", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/my_sdqn", help='directory in which logs should be saved')
parser.add_argument('--info_file', type=str, default="agent/configs_sdqn/default.json", help='path to the file with informations about the model')
args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

file_name_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
file_name = f"{args.log_dir}/{file_name_time}"
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(file_name + ".log")
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

#Get file with informations
info_file = u.get_info_file(args.info_file)
offset_phase_train = info_file['offset_phase_train']
offset_phase = info_file['offset_phase']
flag_default_reward = info_file['flag_default_reward']
flag_mean_reward = info_file['flag_mean_reward']

episodes = args.episodes if info_file['flag_arg_episode'] else info_file['episodes']

#start wandb
u.wand_init("my_sdqn",f'sdqn_{file_name_time}')

# create world
world = World(args.config_file, thread_num=args.thread)

# define tempo de amarelo
yellow_phase_time = 0
# create agents
agents = []
for i in world.intersections:
    yellow_phase_time = i.yellow_phase_time
    action_space = gym.spaces.Discrete(11)
    agents.append(DQNAgent(
        action_space,
        StateOfThreeGenerator(world, i, ["state_of_three"], in_only=True, average=None),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
        i,
        world,
        file_name,
        info_file
    ))
    if args.load_model:
        agents[-1].load_model(args.save_dir)



print(i.phases)

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world)]

# create env
env = TSCEnv(world, agents, metric)

#
n_agents = len(agents)

# train dqn_agent
def train(args, env):
    for e in range(episodes):

        for agent in agents: agent.reset_episode_infos()

        first_obs = np.array(env.reset())*0.01
        current_obs = first_obs

        if e % args.save_rate == args.save_rate - 1:
            env.eng.set_save_replay(True)
            env.eng.set_replay_file("replay_%s.txt" % e)
        else:
            env.eng.set_save_replay(False)

        episodes_rewards = [0] * n_agents
        episodes_decision_num = [0] * n_agents

        i = 0
        while i < args.steps:
            
            ### Requsita nova ação (phase + time) quando acaba o tempo da ação atual
            for agent_id, agent in enumerate(agents):
                agent_obs = current_obs[agent_id]
                if agent.episode_action_time <= i:
                    if agent.episode_action_time == i:
                        agent.change_phase()
                        
                        initial_phase = agent.actual_phase
                        a_phase = initial_phase
                        obs_te = env.world.get_state_of_three_by_phase(agent.I,a_phase)
                        while obs_te[0] == 0:
                            agent.change_phase() 
                            a_phase = agent.actual_phase
                            obs_te = env.world.get_state_of_three_by_phase(agent.I,a_phase)
                            if initial_phase == a_phase:
                                break

                        
                        agent.replay() 
                        #agent.action_time = -1
                        #print(i,agent.get_phase())

                    if agent.episode_action_time+yellow_phase_time+offset_phase <= i:
                        
                        #print(first_obs[agent_id], agent_obs)
                        #print("----")
                        first_obs[agent_id] = agent_obs
                        
                        time = agent.get_action(first_obs[agent_id])
                        agent.action_time = time
                        agent.episode_action_time += (time+1)*5 ## Parte de 0 segundos + tempo decidido pelo modelo (0,5,10,15,20...)
                        phase = agent.I.current_phase
                        #print(i,agent_obs,time,phase,agent.actual_phase)
                        #print(time)

            ### Para cada action interval
            for _ in range(args.action_interval):
                actions = [agent.get_phase() for agent in agents]
                current_obs, current_rewards, dones, current_info = env.step(actions)
                current_obs = np.array(current_obs)*0.01
                i += 1
                
                #u.append_new_line_states(file_name+"_0",[e,i,first_obs,current_obs,[agents[0].get_phase(),agents[0].I.current_phase],[current_rewards[0],agents[0].real_reward(first_obs[0],current_obs[0])]])
                
                for agent_id, agent in enumerate(agents):


                    reward = agent.real_reward(first_obs[agent_id],current_obs[agent_id])
                    #print(reward,current_rewards[agent_id])

                    agent.current_reward.append(current_rewards[agent_id]) if flag_default_reward else agent.current_reward.append(reward) 

                    if agent.episode_action_time+yellow_phase_time+offset_phase == i:
                        action_time = agent.action_time

                        agent_reward = np.mean(agent.current_reward) if flag_mean_reward else agent.current_reward[-yellow_phase_time]
                        #print('----------------')
                        #print("Reward: ", agent_reward,"; min:",np.min(agent.current_reward),"; Méd:",np.mean(agent.current_reward),"; Max:",np.max(agent.current_reward),"; Contagem:",len(agent.current_reward) )
                        #print('----------------')
                        agent.current_reward = []

                        phase = agent.actual_phase
                        next_p = agent.next_phase(phase)

                        u.append_new_line(file_name+f"_{agent_id}",[[first_obs[agent_id],phase], action_time, agent_reward, [current_obs[agent_id],next_p],e,i])
                        ob = first_obs[agent_id].tolist()
                        nob = current_obs[agent_id].tolist()
                        agent.remember( [ob,phase] , action_time, agent_reward, [nob,next_p])
                            
                        episodes_rewards[agent_id] += agent_reward
                        episodes_decision_num[agent_id] += 1

        if agent.total_decision > agent.learning_start:
            agent.decay_epsilon()
            #agent.replay()
            agent.update_target_network()
        #if agent.total_decision > agent.learning_start and not(agent.total_decision%agent.update_target_model_freq) :

        if not (e % args.save_rate):
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            for agent in agents:
                agent.save_model(args.save_dir)
                
        eval_dict = {}

        logger.info(f"episode:{e}/{episodes-1}, steps:{i}")
        eval_dict["episode"]=e
        eval_dict["steps"]=i

        for metric in env.metric:
            logger.info(f"{metric.name}: {metric.eval()}")
            eval_dict[metric.name]=metric.eval()

        for agent_id, agent in enumerate(agents):
            logger.info(f"agent:{agent_id}, epsilon:{agent.epsilon}, mean_episode_reward:{episodes_rewards[agent_id] / episodes_decision_num[agent_id]}")

        eval_dict["epsilon"]=agents[0].epsilon
        eval_dict["mean_episode_reward"]=episodes_rewards[0] / episodes_decision_num[0]
        
        u.wand_log(eval_dict)

    logger.info("Parametros Utilizados")
    agent = agents[0]
    #logger.info(f"BUFFER: buffer_size:{agent.buffer_size}; batch_size:{agent.batch_size}; learning_start:{agent.learning_start};")
    #logger.info(f"MODEL UPDATE: update_model_freq:{agent.update_model_freq}; update_target_model_freq:{agent.update_target_model_freq};")
    #logger.info(f"LEARNING: gamma:{agent.gamma}; epsilon:{agent.epsilon_start}; epsilon_min:{agent.epsilon_min}; epsilon_decay:{agent.epsilon_decay}; learning_rate:{agent.learning_rate};")
    logger.info(f"PHASE: n_phases:{agent.n_phases}; start_phase:{agent.start_phase};")
    logger.info(f"TRAINING: total_decision:{agent.total_decision};")
    #logger.info(f"ACTIVATION: activation:{agent.activation};")
    logger.info(f"STATE: ob_generator:{agent.ob_generator.fns[0]};")
    logger.info(f"REWARD: reward_generator:{agent.reward_generator.fns[0]};")
    logger.info(str(info_file))


def test():
    obs = env.reset()
    for agent in agents:
        agent.load_model(args.save_dir)
    for i in range(args.steps):
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(obs[agent_id]))
        obs, rewards, dones, info = env.step(actions)
        #env.metric.update()
    
        if all(dones):
            break
    logger.info("Final Travel Time is %.4f" % env.metric.update(done=True))


if __name__ == '__main__':
    # simulate
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    train(args, env)
    #test()

import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator, StateOfThreeGenerator
from agent.my_dqn_agent import DQNAgent
from metric import TravelTimeMetric
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
parser.add_argument('--save_dir', type=str, default="model/my_dqn", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/my_dqn", help='directory in which logs should be saved')
args = parser.parse_args()



if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

file_name = f"{args.log_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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

# define tempo de amarelo
yellow_phase_time = 0
# create agents
agents = []
for i in world.intersections:
    yellow_phase_time = i.yellow_phase_time
    action_space = gym.spaces.Discrete(23)
    agents.append(DQNAgent(
        action_space,
        StateOfThreeGenerator(world, i, ["state_of_three"], in_only=True, average=None),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
        i,
        world
    ))
    if args.load_model:
        agents[-1].load_model(args.save_dir)


print(i.phases)
# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

#
n_agents = len(agents)

# train dqn_agent
def train(args, env):
    for e in range(args.episodes):

        for agent in agents: agent.reset_episode_infos()

        first_obs = env.reset()
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
                        #agent.action_time = -1

                    if agent.episode_action_time+yellow_phase_time+1 == i:
                        time = agent.get_action(first_obs[agent_id])
                        first_obs[agent_id] = current_obs[agent_id]
                        agent.action_time = time
                        agent.episode_action_time += 10 + time*2 ## Parte de 15 segundos +  tempo decidido pelo modelo (15,17,19,21,23...)
                    

            ### Para cada action interval
            for _ in range(args.action_interval):
                actions = [agent.get_phase() for agent in agents]
                current_obs, current_rewards, dones, current_info = env.step(actions)
                current_obs = (np.array(current_obs)/100).tolist()
                i += 1
                #u.append_new_line_states(file_name+"_0",[e,i,first_obs,current_obs,agents[0].get_phase(),agents[0].I.current_phase])

                for agent_id, agent in enumerate(agents):
                    agent.current_reward.append(current_rewards[agent_id])

                    if agent.episode_action_time+yellow_phase_time == i:
                        action_time = agent.action_time
                        #agent_reward = agent.current_reward[-1] + agent.current_reward[0]
                        agent_reward = np.mean(agent.current_reward)
                        agent.current_reward = []
                        phase = agent.actual_phase/7
                        next_p = agent.next_phase(agent.actual_phase)/7


                        u.append_new_line(file_name+f"_{agent_id}",[[first_obs[agent_id],phase], action_time, agent_reward, [current_obs[agent_id],next_p],e,i])
                        agent.remember([first_obs[agent_id],phase], action_time, agent_reward, [current_obs[agent_id],next_p])
                            
                        episodes_rewards[agent_id] += agent_reward
                        episodes_decision_num[agent_id] += 1


        if agent.total_decision > agent.learning_start:
            agent.decay_epsilon()
            agent.replay()
            agent.update_target_network()
        #if agent.total_decision > agent.learning_start and not(agent.total_decision%agent.update_target_model_freq) :
            

        if not (e % args.save_rate):
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            for agent in agents:
                agent.save_model(args.save_dir)
        logger.info(f"episode:{e}/{args.episodes}, steps:{i}, average travel time:{env.eng.get_average_travel_time()}")
        for agent_id, agent in enumerate(agents):
            logger.info(f"agent:{agent_id}, mean_episode_reward:{episodes_rewards[agent_id] / episodes_decision_num[agent_id]}")

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

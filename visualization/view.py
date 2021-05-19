   
import numpy as np
from pathlib import Path
import ast 
import matplotlib.pyplot as plt
from datetime import datetime

import seaborn as sns
sns.set_theme(style="darkgrid")

import pandas as pd


import pickle



def get_infos(_path,num_epis=160):
    all_rewards = []
    for epi in range(num_epis): 
        sub_rewards = []
        try:
            f = open(f"{_path}/{epi}_rewards.txt", "r")
        except:
            break
        
        for x in f:
            # Converting string to list 
            res = ast.literal_eval(x)
            sub_rewards.append(res[3])
        
        all_rewards.append(sub_rewards)
    return all_rewards

def plots_rewards(_path,all_rewards):
    steps = []
    rewards = []
    for i,r in enumerate(all_rewards):
        steps.append(i)
        rewards.append(np.mean(r))

    plt.plot(steps,rewards,alpha=1)
    plt.xlabel('step')
    plt.ylabel('rewards')

    plt.savefig(f'{_path}/_reward_plot.png')
    plt.close()

def plots_all_rewards(_path,all_rewards):

    for rewards in all_rewards:
        timefilteredForce = plt.plot(rewards, alpha=0.1)
        timefilteredForce = plt.xlabel('step')
        timefilteredForce = plt.ylabel('rewards')

    plt.savefig(f'{_path}/_all_reward_plot.png')
    plt.close()

def plots_features(_path,name,index,value,xlabel = 'step',ylabel ='cars'):

    plt.plot(value,alpha=1,label="name")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'{_path}/{name}{index}.png')
    plt.close()

def plots_features_and_area(_path,name,index,max_v,mean_v,min_v,xlabel = 'step',ylabel ='cars'):
    
    plt.fill_between(np.arange(start=0, stop=len(mean_v)), min_v, max_v,alpha = 0.4,label="Min and Max!",color='C0')
    plt.plot(mean_v,alpha=1,label="Mean",color='C0')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'{_path}/{name}{index}.png')
    plt.close()

def plots_features_and_area_and_fixedtime(_path,name,index,max_v,mean_v,min_v,fi15,fi30,fi45,xlabel = 'step',ylabel ='cars'):
    
    plt.fill_between(np.arange(start=0, stop=len(mean_v)), min_v, max_v,alpha = 0.4,label="Min and Max!",color='C0')
    plt.plot(mean_v,alpha=1,label="Our Model",color='C0')

    plt.plot(fi15,alpha=.8,label="Fixed 15",linestyle='--',color='C1')
    plt.plot(fi30,alpha=.8,label="Fixed 30",linestyle='--',color='C2')
    plt.plot(fi45,alpha=.8,label="Fixed 45",linestyle='--',color='C3')

    plt.xlabel(xlabel)
    plt.legend()
    plt.ylabel(ylabel)
    plt.savefig(f'{_path}/{name}{index}.png')
    plt.close()

def save_values(string,value):
    f = open(f"{string}.pckl", 'wb')
    pickle.dump(value, f)
    f.close()


def get_infos_log(path):
    average_travel_time = []
    steps_to_end = []
    num_lines = 0
    ## Abrir arquivo
    try:
        f = open(f"{path}/log.log", "r")
    except:
        return []

    ## analisa linha por linha do arquivo
    for line in f:
        num_lines += 1

        ## removemos a parte road, pois ela fica depois do ';' e separa cada item por espaço
        sub_line = line.split(';')[0]
        ## se não tiver nenhum carro, pula
        if (len(sub_line) == 0):
            continue
        
        sub_line = sub_line.split(',')
        if(sub_line[-1][1:8]=="average"):
            steps = float(sub_line[1].split(":")[-1][:])
            att = float(sub_line[-1].split(":")[-1][:])
            steps_to_end.append(steps)
            average_travel_time.append(att)

    f.close()
    
    return average_travel_time,steps_to_end

_path = f"mydqn/"

paths = [_path,_path]
print(paths)
average_time_travel = []
total_time = []

for path in paths:
    att, steps = get_infos_log(path)
    average_time_travel.append(att)
    total_time.append(steps)

path = "mydqn/"

min_att,mean_att,max_att = [],[],[]
min_tt,mean_tt,max_tt = [],[],[]

average_time_travel = pd.DataFrame(average_time_travel)
total_times = pd.DataFrame(total_time)

for a in range(len(average_time_travel.iloc[0])):
    min_att.append(np.mean(average_time_travel[a]) - np.std(average_time_travel[a]))
    mean_att.append(np.mean(average_time_travel[a]))
    max_att.append(np.mean(average_time_travel[a]) + np.std(average_time_travel[a]))

    min_tt.append(np.mean(total_times[a]) - np.std(total_times[a]))
    mean_tt.append(np.mean(total_times[a]))
    max_tt.append(np.mean(total_times[a]) + np.std(total_times[a]))

save_values(f"{path}/tt",[min_tt,mean_tt,max_tt])
save_values(f"{path}/att",[min_att,mean_att,max_att])


#plots_features_and_area_and_fixedtime(path,"num_car","",max_mnc,mean_mnc,min_mnc, fixed_info_15[0], fixed_info_30[0], fixed_info_45[0], xlabel = 'step',ylabel ='cars')
#plots_features_and_area_and_fixedtime(path,"time_travel","",max_mtt,mean_mtt,min_mtt, fixed_info_15[1], fixed_info_30[1], fixed_info_45[1], xlabel = 'step',ylabel ='seconds')
#plots_features_and_area_and_fixedtime(path,"time_total","",max_tt,mean_tt,min_tt, fixed_info_15[2], fixed_info_30[2], fixed_info_45[2], xlabel = 'step',ylabel ='seconds')
#plots_features_and_area_and_fixedtime(path,"average_vel","",max_av,mean_av,min_av, fixed_info_15[3], fixed_info_30[3], fixed_info_45[3], xlabel = 'step',ylabel ='vel')
#plots_features_and_area_and_fixedtime(path,"average_waiting","",max_aw,mean_aw,min_aw, fixed_info_15[4], fixed_info_30[4], fixed_info_45[4], xlabel = 'step',ylabel ='seconds')


plots_features_and_area(path,"time_travel","",max_att,mean_att,min_att, xlabel = 'step',ylabel ='seconds')
plots_features_and_area(path,"time_total","",max_tt,mean_tt,min_tt, xlabel = 'step',ylabel ='seconds')

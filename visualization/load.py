   
import numpy as np
from pathlib import Path
import ast 
import matplotlib.pyplot as plt
from datetime import datetime

import seaborn as sns
sns.set_theme(style="darkgrid")

import pandas as pd

import pickle

def plots_features_and_area_and_fixedtime(_path,name,index,max_v,mean_v,min_v,fi15,fi30,fi45,xlabel = 'step',ylabel ='cars',position='best'):
    
    plt.fill_between(np.arange(start=0, stop=len(mean_v)), min_v, max_v,alpha = 0.4,color='C0')
    

    plt.plot(fi15,alpha=.8,label="FT15",linestyle='--',color='C1')
    plt.plot(fi30,alpha=.8,label="FT30",linestyle='--',color='C2')
    plt.plot(fi45,alpha=.8,label="FT45",linestyle='--',color='C3')
    plt.plot(mean_v,alpha=1,label="Our Model",color='C0')

    plt.xlabel(xlabel)
    plt.legend(fancybox=True,loc=position)
    plt.ylabel(ylabel)
    plt.savefig(f'{_path}/{name}{index}.png')
    plt.close()

def plots_features(_path,name,index,value,xlabel = 'step',ylabel ='cars'):

    plt.plot(value,alpha=1,label="name")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'{_path}/{name}{index}.png')
    plt.close()

def save_values(string,value):
    f = open(f"{string}.pckl", 'wb')
    pickle.dump(value, f)
    f.close()

def load_values(string):
    f = open(string+".pckl", 'rb')
    shap_values = pickle.load(f)
    f.close()
    return shap_values

path = "DQN"

min_mtt,mean_mtt,max_mtt = load_values(f"{path}/mtt")
min_mnc,mean_mnc,max_mnc = load_values(f"{path}/mnc")
min_tt,mean_tt,max_tt = load_values(f"{path}/tt")
min_av,mean_av,max_av = load_values(f"{path}/av")
min_aw,mean_aw,max_aw = load_values(f"{path}/aw")



print(mean_mnc[-1], " +- ", min_mnc[-1]-mean_mnc[-1], max_mnc[-1]-mean_mnc[-1]) 
print(mean_mtt[-1], " +- ", min_mtt[-1]-mean_mtt[-1], max_mtt[-1]-mean_mtt[-1]) 
print(mean_tt[-1], " +- ", min_tt[-1]-mean_tt[-1], max_tt[-1]-mean_tt[-1]) 
print(mean_av[-1], " +- ", min_av[-1]-mean_av[-1], max_av[-1]-mean_av[-1]) 
print(mean_aw[-1], " +- ", min_aw[-1]-mean_aw[-1], max_aw[-1]-mean_aw[-1]) 


#plots_features_and_area_and_fixedtime(path,"num_car","",max_mnc,mean_mnc,min_mnc, fixed_info_15[0], fixed_info_30[0], fixed_info_45[0], xlabel = 'step',ylabel ='vehicles',position='lower right')
#plots_features_and_area_and_fixedtime(path,"time_travel","",max_mtt,mean_mtt,min_mtt, fixed_info_15[1], fixed_info_30[1], fixed_info_45[1], xlabel = 'step',ylabel ='seconds',position='upper right')
#plots_features_and_area_and_fixedtime(path,"time_total","",max_tt,mean_tt,min_tt, fixed_info_15[2], fixed_info_30[2], fixed_info_45[2], xlabel = 'step',ylabel ='seconds',position='upper right')
#plots_features_and_area_and_fixedtime(path,"average_vel","",max_av,mean_av,min_av, fixed_info_15[3], fixed_info_30[3], fixed_info_45[3], xlabel = 'step',ylabel ='speed',position='lower right')
#plots_features_and_area_and_fixedtime(path,"average_waiting","",max_aw,mean_aw,min_aw, fixed_info_15[4], fixed_info_30[4], fixed_info_45[4], xlabel = 'step',ylabel ='seconds',position='upper right')

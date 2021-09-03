import numpy
from itertools import product

import json 

possibilits = {
        "validation": [True],
        "episodes": [200],
        "pre_train": [False],
        "action_interval": [5, 10, 20],
        "gamma":[0.1, 0.5, 0.995],
        "epsilon_initial":[0.1, 1],
        "epsilon_min":[0.0, 0.01],
        "epsilon_decay":[0.97, 0.995] ,
        "batch_size":[32,512],
        "buffer_size":[1000, 10000],
        "learning_start":[1000, 5000],
        "update_model_freq":[1,20],
        "update_target_model_freq":[1,10],
        "epochs_replay":[1,5],
        "epochs_initial_replay":[2000],
        "hiden_layers": [1,5],
        "hiden_nodes":[20],
        "hiden_activation":["relu"],
        "output_activation":["relu"],
        "optimizer":["rmsprop"],
        "loss":["mse"]
    }


n_possibilidades = 1
for pos in possibilits.keys():
    n_possibilidades = n_possibilidades*len(possibilits[pos])

#print(n_possibilidades)

num = len(possibilits.values())
keys = list(possibilits.keys())
num_keys = len(keys)
values = list(possibilits.values())

final_list = list(product(*values))

print(len(final_list))

for i,value in enumerate(final_list):
    new_dict = {}
    for ii,v in enumerate(value):
        new_dict[keys[ii]] = v
    
    with open(f"scripts/files/{i}.json", "w") as outfile:
        json.dump(new_dict, outfile)

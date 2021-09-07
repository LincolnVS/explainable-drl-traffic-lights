import numpy
from itertools import product

import json 

possibilits = {
    "validation":[True],

    "episodes":[200],
    "pre_train":[False],
    "action_interval":[20],

    "gamma":[0.9],
    "epsilon_initial":[1],
    "epsilon_min":[0.01],
    "epsilon_decay":[0.999],

    "batch_size":[32,128],
    "buffer_size":[5000],
    "learning_start":[2000],
    "update_model_freq":[1],
    "update_target_model_freq":[20],
    "epochs_replay":[1],
    "epochs_initial_replay":[2000],

    "hiden_layers":[3,5],
    "hiden_nodes":[20,50],
    "hiden_activation":["relu"],
    "output_activation":["linear"],
    "optimizer":["rmsprop","adam"],
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

#!/bin/bash

file_name="log/dqn_jinan_3x4_all.log"

info_file="agent/configs_new_dqn4/"
dataset_name="envs/jinan_3x4/config_dqn.json"

min_test=0
max_test=16

echo "...Starting all new_dqn..." | tee $file_name
for i in $(seq $min_test $max_test)
do
   echo "Running $i/$max_test, $info_file$i.json"  | tee -a $file_name
   python run_new_dqn_phase.py $dataset_name --parameters $info_file$i".json" | tee -a $file_name
   printf "\n" | tee -a $file_name
done   

python visualization/view-all.py "log/my_new_dqn/" | tee -a $file_name


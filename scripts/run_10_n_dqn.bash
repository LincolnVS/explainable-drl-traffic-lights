#!/bin/bash

file_name="log/my_new_dqn_run.log"

info_file="agent/configs_new_dqn4/"
dataset_name="envs/real_1x1/config_cluster3.json"

min_test=1
max_test=11

echo "...Starting all sdqn..." | tee $file_name
for i in $(seq $min_test $max_test)
do
   echo "Running $i/$max_test, 8.json"  | tee -a $file_name
   python run_new_dqn.py $dataset_name --parameters $info_file"8.json" | tee -a $file_name
   printf "\n" | tee -a $file_name
done   

python visualization/view-all.py "log/my_new_dqn/" | tee -a $file_name


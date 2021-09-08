#!/bin/bash

file_name="log/my_new_dqn_run.log"

info_file="agent/configs_new_dqn3/"
dataset_name="envs/real_1x1/config_cluster.json"

min_test=1
max_test=17

echo "...Starting all sdqn..." | tee $file_name
for i in $(seq $min_test $max_test)
do
   echo "Running $i/$max_test, $info_file$i.json"  | tee -a $file_name
   python run_new_dqn.py $dataset_name --parameters $info_file$i".json" | tee -a $file_name
   printf "\n" | tee -a $file_name
done   

python visualization/view-all.py "log/my_new_dqn/" | tee -a $file_name


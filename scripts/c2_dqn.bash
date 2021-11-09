#!/bin/bash

file_name="log/dqn_jinan_3x4.log"

info_file="agent/configs_new_dqn6/"
dataset_name="envs/jinan_3x4/config_dqn.json"

min_test=1
max_test=30

echo "...Starting all sdqn..." | tee $file_name
for i in $(seq $min_test $max_test)
do
   echo "Running $i/$max_test, default.json"  | tee -a $file_name
   python run_new_dqn_phase.py $dataset_name --parameters $info_file"default.json" | tee -a $file_name
   printf "\n" | tee -a $file_name
done   

python visualization/view-all.py "log/my_new_dqn/" | tee -a $file_name


#!/bin/bash

file_name="log/my_dqn_run.log"

info_file="configs_mdqn/"
dataset_name="datasets/real_1x1/config_4p.json"

min_test=1
max_test=15

echo "...Starting all mydqn..." | tee $file_name
for i in $(seq $min_test $max_test)
do
   echo "Running $i/$max_test, $info_file$i.json"  | tee -a $file_name
   python run_my_dqn.py $dataset_name --info_file $info_file$i".json" | tee -a $file_name
   python visualization/view.py log/my_dqn/ | tee -a $file_name
   printf "\n" | tee -a $file_name
done
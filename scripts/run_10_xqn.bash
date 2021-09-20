#!/bin/bash

file_name="log/xqn_run.log"

info_file="agent/configs_xqn/"
dataset_name="envs/real_1x1/config_cluster.json"

min_test=1
max_test=11

echo "...Starting all xqn..." | tee $file_name
for i in $(seq $min_test $max_test)
do
   echo "Running $i/$max_test, default.json"  | tee -a $file_name
   python run_xqn.py $dataset_name | tee -a $file_name
   printf "\n" | tee -a $file_name
done   

python visualization/view-all.py "log/xqn/" | tee -a $file_name


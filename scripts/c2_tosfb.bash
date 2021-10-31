#!/bin/bash

file_name="log/tosfb_jinan_3x4.log"

dataset_name="envs/jinan_3x4/config_tosfb.json"

min_test=1
max_test=30

echo "...Starting 10 tosfb..." | tee $file_name
for i in $(seq $min_test $max_test)
do
   echo "Running $i/$max_test"  | tee -a $file_name
   python run_n_tosfb.py $dataset_name | tee -a $file_name
   printf "\n" | tee -a $file_name
done   

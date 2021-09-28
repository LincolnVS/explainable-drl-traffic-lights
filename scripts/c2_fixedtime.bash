#!/bin/bash

file_name="log/fixedtime/jinan_3x4.log"
dataset_name="envs/jinan_3x4/config_fixedtime.json"

min_time=1
max_time=60

echo "...Starting all fixedtime..." | tee $file_name
for i in $(seq $min_time $max_time)
do
   echo "Running $i/$max_time" | tee -a $file_name
   python run_fixedtime.py $dataset_name --phase_time $i | tee -a $file_name
   printf "\n" | tee -a $file_name
done
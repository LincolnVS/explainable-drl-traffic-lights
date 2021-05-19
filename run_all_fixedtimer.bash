#!/bin/bash

file_name="log/fixedtime/real1x1.log"

echo "...Starting all fixedtime..." | tee $file_name
for i in {1..56}
do
   echo "Running $i/56" | tee -a $file_name
   python run_fixedtime.py datasets/real_1x1/config.json --phase_time $i | tee -a $file_name
   printf "\n" | tee -a $file_name
done
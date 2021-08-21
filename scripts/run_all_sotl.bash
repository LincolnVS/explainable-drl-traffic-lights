min_green_time=(10 15 20 25 30)
green_v=(10 15 20 25 30)
red_v=(20 25 30 35 40)

file_name="log/sotl/real1x1.log"
dataset_name="envs/real_1x1/config_cluster.json"

echo "...Starting all sotl possibilities..." | tee $file_name
for mgt in "${min_green_time[@]}"
do
	for gv in "${min_green_time[@]}"
    do
        for rv in "${min_green_time[@]}"
        do
            echo "$mgt $gv $rv" | tee -a $file_name
            python run_sotl.py $dataset_name --green_time $mgt --green_v $gv --red_v $rv | tee -a $file_name
            printf "\n" | tee -a $file_name
        done
    done
done
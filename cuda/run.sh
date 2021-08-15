#!/bin/bash

set -o errexit
set -o xtrace

now=$(date +"%Y%m%d_%H%M%S")
#slurm_command="srun -p gpu_volta --gres=gpu:1"
#slurm_command="gdb --args"
slurm_command=""

plan=5
plan_folder=~/research/Radiotherapy/plans/$plan

results_folder=results
result_path=$results_folder/x_$plan_$now.txt
mkdir -p $results_folder

fluence_folder=~/research/Radiotherapy/fluences
fluence_prefix=x_5_start_0001

$slurm_command ./gradient $plan $plan_folder $result_path $fluence_folder $fluence_prefix

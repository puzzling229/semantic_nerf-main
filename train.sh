#!/bin/bash

echo GPU $1
export CUDA_VISIBLE_DEVICES=$1
log=./log_nohup/sem_NeRF_time_log.out
config_name=./SSR/configs/SSR_room0_config.yaml
nohup python3 train_SSR_main.py --config ${config_name} > ${log} 2>&1 &
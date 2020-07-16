#!/bin/bash

path=/private/home/wenboren/datasets/cifar-10-batches-py
local_steps=1
initial_steps=100
# initial_step_method=single_process
initial_step_method=multiple_processes

python cifar.py $path \
    --dist-url 'tcp://127.0.0.1:23453' --dist-backend 'nccl' \
    -p 10 --epochs 5 --batch-size 256 \
    --local-steps $local_steps \
    --initial-steps $initial_steps \
    --initial-step-method $initial_step_method \
    | tee cifar.log

#!/bin/bash

path=~/datasets/imagenet
local_steps=4
initial_steps=100
initial_step_method=single_process
# initial_step_method=multiple_processesrint("")
       
python imtest.py $path -a resnet18 \
    --dist-url 'tcp://127.0.0.1:23451' --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 --rank 0 --epochs 2 \
    --local-sgd --local-steps $local_steps \
    --initial_steps $initial_steps \
    --initial_step_method $initial_step_method \
    | tee imagenet.log

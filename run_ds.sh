#!/bin/bash

deepspeed \
    --num_gpus 8 \
    --num_nodes 2 \
    --hostfile hostfile \
    --master_addr 10.252.32.12 \
    randy/train.py "$@"

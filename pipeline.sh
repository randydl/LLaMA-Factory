#!/bin/bash

./train.sh randy/qwen2_pt3.yaml
sleep 900

./train.sh randy/qwen2_pt4.yaml
sleep 900

./train.sh randy/qwen2_pt5.yaml

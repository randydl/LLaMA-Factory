import os
import fire
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
from llamafactory.train.tuner import run_exp


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://172.19.92.23:13128'
os.environ['https_proxy'] = 'http://172.19.92.23:13128'


if __name__ == '__main__':
    args = OmegaConf.load('examples/train_lora/llama3_lora_pretrain.yaml')
    args = OmegaConf.merge(args, {
        'model_name_or_path': '/nas_data/userdata/randy/models/Meta-Llama-3-8B',
        'quantization_bit': 4
    })
    run_exp(args)

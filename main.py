import time
import subprocess
from loguru import logger


base_cmd = (
    'deepspeed '
    '--hostfile randy/hostfile '
    'randy/train.py '
)


def run_task(*args, **kwargs):
    try:
        kwargs = [f'--{k}={v}' for k, v in kwargs.items()]
        full_cmd = f'{base_cmd}{" ".join(list(args) + kwargs)}'
        subprocess.run(full_cmd.split(), check=True)
    except Exception as e:
        logger.error(e)


if __name__ == '__main__':
    logger.add('randy/train.log', mode='w')
    run_task('randy/qwen2_pt.yaml')
    run_task('randy/qwen2_pt.yaml')
    run_task('')

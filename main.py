import subprocess
from loguru import logger


base_cmd = (
    'deepspeed '
    '--num_gpus 8 '
    '--num_nodes 2 '
    '--hostfile hostfile '
    '--master_addr 10.252.32.12 '
    'randy/train.py '
)


def run_task(additional_args):
    try:
        full_cmd = f'{base_cmd}{additional_args}'
        subprocess.run(full_cmd.split(), check=True)
    except Exception as e:
        logger.error(e)


if __name__ == '__main__':
    logger.add('randy/train.log')
    run_task('')  # Example call with no additional arguments

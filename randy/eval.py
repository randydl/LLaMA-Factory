import os
import fire
import subprocess
from pathlib import Path
from llamafactory.eval.evaluator import Evaluator


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://172.19.92.23:13128'
os.environ['https_proxy'] = 'http://172.19.92.23:13128'


def main(model_path, task, template='fewshot'):
    model_path = Path(model_path)
    assert task in ['mmlu', 'cmmlu']

    save_dir = model_path.joinpath(task)
    lang = {'mmlu': 'en', 'cmmlu': 'zh'}[task]
    subprocess.run(f'rm -rf {save_dir}', shell=True)

    evaluator = Evaluator({
        'model_name_or_path': str(model_path),
        'save_dir': str(save_dir),
        'task': f'{task}_test',
        'lang': lang,
        'template': template,
        'n_shot': 5,
        'batch_size': 1
    })
    evaluator.eval()


if __name__ == '__main__':
    fire.Fire(main)


# pip install --no-deps -e .
# CUDA_VISIBLE_DEVICES=6 python randy/eval.py --task=mmlu --model_path=""
# CUDA_VISIBLE_DEVICES=7 python randy/eval.py --task=cmmlu --model_path=""
# CUDA_VISIBLE_DEVICES=6 python randy/eval.py --task=mmlu --template=llama3 --model_path=""
# CUDA_VISIBLE_DEVICES=7 python randy/eval.py --task=cmmlu --template=llama3 --model_path=""

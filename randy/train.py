import sys
from omegaconf import OmegaConf


def parse_args(argv):
    args = []
    for arg in argv:
        if not arg.startswith('-') and arg.endswith('.yaml'):
            conf = OmegaConf.load(arg)
            for k, v in conf.items():
                args.append(f'--{k}={v}')
        else:
            args.append(arg)
    return args


if __name__ == '__main__':
    args = ' '.join(parse_args(sys.argv[1:]))
    print(f'<randy>src/train.py {args}</randy>')

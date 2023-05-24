import yaml
from easydict import EasyDict as edict

def load_config(path: str):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)
    return cfg
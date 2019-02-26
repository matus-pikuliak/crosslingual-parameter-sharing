"""
python evaluate.py RUN_NAME [CONFIG_OPTIONS]
"""
import ast
import os
import sys

from config.config import Config
from data.data_loader import DataLoader
from model.model import Model


def get_representations(run_name):
    pth = f'/home/fiit/oink/'
    config = Config.load_from_log(run_name, pth)
    config.values['log_path'] = pth
    config.values['model_path'] = pth
    dl = DataLoader(config)
    dl.load()

    with Model(dl, config, name=run_name) as model:
        model.load(f'{run_name}-1')
        return model.get_representations()


if __name__ == '__main__':
    print(get_representations(run_name=sys.argv[1]))

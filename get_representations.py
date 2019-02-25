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
    config = Config.load_from_log(run_name)
    dl = DataLoader(config)
    dl.load()

    with Model(dl, config, name=run_name) as model:
        return model.get_representations()


if __name__ == '__main__':
    print(get_representations(run_name=sys.argv[1]))

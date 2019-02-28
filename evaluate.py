"""
python evaluate.py RUN_NAME [CONFIG_OPTIONS]
"""

import sys

from config.config import Config
from data.data_loader import DataLoader
from model.model import Model


def evaluate(run_name, config):
    dl = DataLoader(config)
    dl.load()

    with Model(dl, config, name=run_name) as model:
        model.run_evaluation()


if __name__ == '__main__':
    config = Config(*sys.argv[2:])
    evaluate(run_name=sys.argv[1], config=config)

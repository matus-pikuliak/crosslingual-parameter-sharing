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

    model = Model(dl, config, name=run_name)
    model.build_graph()
    model.run_evaluation()
    model.close()


if __name__ == '__main__':
    config = Config(*sys.argv[2:])
    evaluate(run_name=sys.argv[1], config=config)

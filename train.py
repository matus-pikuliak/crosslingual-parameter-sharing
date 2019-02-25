"""
python train.py [CONFIG_OPTIONS]
"""

import sys

from config.config import Config
from data.data_loader import DataLoader
from model.model import Model

config = Config(*sys.argv[1:])

dl = DataLoader(config)
dl.load()

with Model(dl, config) as model:
    model.run_experiment()

"""
python evaluate.py RUN_NAME [CONFIG_OPTIONS]
"""

import sys

from config.config import Config
from data.data_loader import DataLoader
from model.model import Model

config = Config(*sys.argv[2:])

dl = DataLoader(config)
dl.load()

model = Model(dl, config, name=sys.argv[1])
model.build_graph()
model.run_evaluation()
model.close()

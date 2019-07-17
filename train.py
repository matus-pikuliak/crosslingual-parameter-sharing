"""
python train.py [CONFIG_OPTIONS]
"""

import sys

from config.config import Config
from data.data_loader import DataLoader
from model.orchestrator import Orchestrator

config = Config(*sys.argv[1:])

dl = DataLoader(config)
dl.load()

with Orchestrator(dl, config) as orch:
    orch.run_training()

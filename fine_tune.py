"""
python continue_train.py RUN_NAME FINE_TUNE_TASK
"""
import ast
import os
import sys

from config.config import Config
from data.data_loader import DataLoader
from model.model import Model

run_name = sys.argv[1]
default_config = Config()
log_file_path = os.path.join(default_config.log_path, run_name)
with open(log_file_path) as log_file:
    log = ast.literal_eval(log_file.read())
    config = Config.load_from_dict(log['config'])

dl = DataLoader(config)
dl.load()
config.values['train_only'] = sys.argv[2]
max_epoch = config.values['epoch_steps']
config.values['epochs'] += int(sys.argv[3])

with Model(dl, config) as model:
    model.load(f'{run_name}-{max_epoch}')
    model.run_experiment(start_epoch=max_epoch+1)

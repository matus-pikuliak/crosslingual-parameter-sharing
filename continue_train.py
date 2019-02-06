"""
python continue_train.py RUN_NAME [CONFIG_OPTIONS]
"""
import ast
import os
import sys

from config.config import Config
from data.data_loader import DataLoader
from model.model import Model

run_name = sys.argv[1]
default_config = Config(*sys.argv[2:])
log_file_path = os.path.join(default_config.log_path, run_name)
with open(log_file_path) as log_file:
    log = ast.literal_eval(log_file.read())
    config = Config()
    config.load_from_dict(log['config'])

dl = DataLoader(config)
dl.load()
model = Model(dl, config)
model.build_graph()

max_epoch = max(record['epoch'] for record in log['results'])

try:
    assert(sum(1 for record in log['results'] if record['epoch'] == max_epoch) == len(config.tasks) * 3)
    model.load(f'{run_name}-{max_epoch}')
except:
    model.load(f'{run_name}-{max_epoch-1}')
    log['result'] = [record for record in log['results'] if record['epoch'] != max_epoch]
    max_epoch -= 1

if config.setup == 'production':
    model.logger.results = log
model.run_experiment(start_epoch=max_epoch+1)
model.close()

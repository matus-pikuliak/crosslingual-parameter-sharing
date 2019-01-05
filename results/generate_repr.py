"""
This script generates word level representations for saved model. As such we need a log file (for config reasons) and
tensorflow saver file. Both are identified by the run id.
"""

import ast
import glob
import itertools
import os
import sys

sys.path.append(os.path.abspath('..'))
from config.config import Config
from data.data_loader import DataLoader
from model.model import Model

# for logfile_path in glob.glob(f'{Config().log_path}gcp/*'):
for logfile_path in glob.glob(f'{Config().log_path}gcp/2018-12-13-151057'):

    if not os.path.isfile(logfile_path):
        continue

    with open(logfile_path) as logfile:
        for line in logfile:
            if 'optimizer' in line:
                break
        else:
            continue

        config_values = ast.literal_eval(line[1:].strip())

        config = Config()
        config.load_from_logfile(config_values)

        dl = DataLoader(config)
        dl.load()

        timestamp = os.path.split(logfile_path)[-1]

        for modelfile_path in glob.glob(f'{Config().model_path}{timestamp}*.index'):

            print(modelfile_path)

            model_id = os.path.split(modelfile_path)[-1]
            model_id = model_id.split('.index')[0]
            config.values['load_model'] = model_id

            model = Model(dl, config)
            model.build_graph()

            for (task, lang), role in itertools.product(config.tasks, ('test', 'train')):
                model.temp_export_representations(task, lang, role, sample_size=500)

            model.close()

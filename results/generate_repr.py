"""
This script generates word level representations for saved model. As such we need a log file (for config reasons) and
tensorflow saver file. Both are identified by the run id.
"""

import ast
import glob
import h5py
import itertools
import os
import sys

sys.path.append(os.path.abspath('..'))
from config.config import Config
from data.data_loader import DataLoader
from model.model import Model

for logfile_path in glob.glob(f'{Config().log_path}{sys.argv[1]}/*'):

    if not os.path.isfile(logfile_path):
        continue

    with open(logfile_path) as logfile:
        print(f'Wprking on {logfile_path}')
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

            model_id = os.path.split(modelfile_path)[-1]
            model_id = model_id.split('.index')[0]
            config.values['load_model'] = model_id

            model = Model(dl, config)
            model.build_graph()

            for (task, lang), role in itertools.product(config.tasks, ('test', 'train')):
                output = model.temp_export_representations(task, lang, role)
                output_file_path = f'{modelfile_path}-{task}-{lang}-{role}.h5'
                if not os.path.isfile(output_file_path):
                    with h5py.File(output_file_path, 'w') as f:
                        for k, v in output.items():
                            f.create_dataset(name=k, data=v)
                    print(f'{output_file_path} was created')
                else:
                    print(f'{output_file_path} found. Skipped')

            model.close()

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

subdir = ''
try:
    subdir = sys.argv[1]
except IndexError:
    pass

for logfile_path in glob.glob(f'{Config().log_path}{subdir}*'):

    if not os.path.isfile(logfile_path):
        continue

    with open(logfile_path) as logfile:
        print(f'Wprking on {logfile_path}')
        for line in logfile:
            if 'optimizer' in line:
                break
        else:
            continue

        data_loaded = False

        config_values = ast.literal_eval(line[1:].strip())

        config = Config()
        config.load_from_logfile(config_values)


        timestamp = os.path.split(logfile_path)[-1]

        for modelfile_path in glob.glob(f'{Config().model_path}{timestamp}*.index'):
            model_loaded = False

            model_id = os.path.split(modelfile_path)[-1]
            model_id = model_id.split('.index')[0]
            config.values['load_model'] = model_id
            config.values['setup'] = 'default'


            for (task, lang), role in itertools.product(config.tasks, ('test', 'train')):
                output_file_path = os.path.join(Config().model_path, f'{model_id}-{task}-{lang}-{role}.h5')
                if not os.path.isfile(output_file_path):

                    if not data_loaded:
                        dl = DataLoader(config)
                        dl.load()
                        data_loaded = True

                    if not model_loaded:
                        model = Model(dl, config)
                        model.build_graph()
                        model_loaded = True

                    output = model.temp_export_representations(task, lang, role)
                    with h5py.File(output_file_path, 'w') as f:
                        for k, v in output.items():
                            f.create_dataset(name=k, data=v)
                    print(f'{output_file_path} was created')
                else:
                    print(f'{output_file_path} found. Skipped')

            model.close()

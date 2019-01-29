import ast
import glob
import os
import sys

sys.path.append(os.path.abspath('..'))
from config.config import Config

old_log_path = sys.argv[1]
model_path = ...
new_log_path = ...

for log_file in glob.glob(os.path.join(old_log_path, '*')):
    if os.path.isfile(log_file):
        with open(log_file) as log_f:
            contents = ast.literal_eval(log_f.read())
            config = Config()
            config.load_from_logfile(contents['hparams'])
            print(config.values)
    # import hparams
    # change paths
    # load models and run evaluation with them


import ast
import glob
import os
import sys

sys.path.append(os.path.abspath('..'))
from config.config import Config
from evaluate import evaluate

old_log_path = sys.argv[1]
model_path = sys.argv[2]
new_log_path = sys.argv[3]

for log_file in glob.glob(os.path.join(old_log_path, '*')):
    if os.path.isfile(log_file):
        _, run_name = os.path.split(log_file)
        with open(log_file) as log_f:
            contents = ast.literal_eval(log_f.read())
            config = Config()
            config.load_from_dict(contents['hparams'])
            config.values['log_path'] = new_log_path
            config.values['model_path'] = model_path
            evaluate(run_name, config)

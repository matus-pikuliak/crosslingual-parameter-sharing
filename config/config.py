import ast
import os

import yaml

from constants import TASKS, LANGS


class Config:
    """
    Config stores all the hyperparameters for the model. The procedure is as follows:

    1. Load default parameters from hparams and private.yaml files.
    2. If the first user argument is 'setup', overwrite values by setup-specific parameters from both
       hparams and private.
    3. Overwrite loaded parameter values with those specified in args in "key value" format.
    4. The last optional user argument is tasks followed by "task-language" arguments.

    Each parameter value is type checked against the types specified in yaml files. Boolean values
    are interpreted as True if the value is string "true" in any casing, otherwise they are
    interpreted as False. String parameters that are supposed to be None are set as "na" in the
    yaml files. All the specified parameters must already be in yaml files.
    """

    def __init__(self, *args):
        dir = os.path.dirname(__file__)
        hparams = yaml.safe_load(open(os.path.join(dir, 'hparams.yaml')))
        settings = yaml.safe_load(open(os.path.join(dir, 'private.yaml')))

        self.values = hparams['default']
        self.values.update(settings['default'])

        ite = enumerate(args)
        for i, arg in ite:

            if arg not in self.values:
                raise AttributeError('Argument \'%s\' is not permitted.' % arg)

            if arg == 'tasks':
                self.values['tasks'] = [tuple(tl.split('-')) for _, tl in ite]
                if self.values['tasks'] == [('all', )]:
                    self.values['tasks'] = [(task, lang) for task in TASKS for lang in LANGS]
                break

            _, value = next(ite)

            if arg == 'setup':
                if i > 0:
                    raise AttributeError('Argument \'setup\' must be first.')
                self.values.update(hparams[value])
                self.values.update(settings[value])

            arg_type = type(self.values[arg])
            try:
                if arg_type == bool:
                    value = (value.lower() == 'true')
                else:
                    value = arg_type(value)
            except:
                raise AttributeError('Could not type value of %s to %s' % (arg, arg_type))

            self.values[arg] = value

        for k, v in self.values.items():
            if v == 'na':
                self.values[k] = None

    def __getattr__(self, item):
        return self.values[item]

    def __repr__(self):
        return str(self.values)

    @staticmethod
    def load_from_dict(dct):
        config = Config()
        config.values.update(dct)
        dir = os.path.dirname(__file__)
        settings = yaml.safe_load(open(os.path.join(dir, 'private.yaml')))
        config.values.update(settings['default'])
        return config

    @staticmethod
    def load_from_log(run_name=None, log_path=None):
        if run_name is not None:
            log_path = os.path.join(Config().log_path, run_name)
        with open(log_path) as log_file:
            log = ast.literal_eval(log_file.read())
            config = Config.load_from_dict(log['config'])
            return config

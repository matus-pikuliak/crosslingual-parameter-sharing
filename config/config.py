import os

import yaml


class Config:
    '''
    Config stores all the hyperparameters and settings for the model. The procedure is as follows:

    1. Load default parameters from hparams and settings.yaml.
    2. If the first user argument is 'setup', overwrite values by setup-specific values from both
       hparams and settings.
    3. Overwrite loaded values with those specified in args in "key value" format.
    (Not supported yet) X. If load_model is specified and load_settings is set to true, load config
                           values from log file.
    5. The last user argument is tasks followed by "task-language" arguments.

    Each parameter value is type checked against the types specified in yaml files. Boolean values
    are interpreted as True if the value is string "true" in any casing, otherwise they are
    interpreted as False. String parameters that are supposed to be None are set as "na" in the
    yaml files. All the specified parameters must already be in yaml files.
    '''

    #TODO: treba zlepsit wording tohoto popisku, aby bolo jasne co je argument / parameter / yaml files

    def __init__(self, *args):
        dir = os.path.dirname(__file__)
        hparams = yaml.safe_load(open(os.path.join(dir, 'hparams.yaml')))
        settings = yaml.safe_load(open(os.path.join(dir, 'settings.yaml')))

        self.values = hparams['default']
        self.values.update(settings['default'])

        ite = enumerate(args)
        for i, arg in ite:

            if arg not in self.values:
                raise AttributeError('Argument \'%s\' is not permitted.' % arg)

            if arg == 'tasks':
                self.values['tasks'] = [tuple(tl.split('-')) for _, tl in ite]
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

            if value == 'na':
                value = None

            self.values[arg] = value

    def __getattr__(self, item):
        return self.values[item]

    def __repr__(self):
        return str(self.values)

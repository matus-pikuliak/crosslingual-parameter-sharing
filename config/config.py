from logs.debug_logger import DebugLogger
from logs.default_logger import DefaultLogger
from logs.production_logger import ProductionLogger
import yaml
import os


class Config:

    def __init__(self, args=[]):

        # setup must be first argument
        # tasks must be last argument in 'task-lang task-lang' format
        # anything between must be already contained in yaml's and must match the type

        dir = os.path.dirname(__file__)
        hparams = yaml.safe_load(file(os.path.join(dir, 'hparams.yaml'), 'r'))
        settings = yaml.safe_load(file(os.path.join(dir, 'settings.yaml'), 'r'))
        values = hparams['default']
        values.update(settings['default'])

        skip_next = False
        for i, arg in enumerate(args):

            if skip_next:
                skip_next = False
                continue

            if arg == 'tasks':
                values['tasks'] = [tuple(a.split('-')) for a in args[i+1:]]
                break

            if arg == 'setup' and i != 0:
                raise AttributeError('Argument \'setup\' must be first.')

            if arg == 'setup':
                setup = args[i+1]
                values.update(hparams[setup])
                values.update(settings[setup])

            value = args[i+1]
            if arg not in values:
                raise AttributeError('Argument \'%s\' is not permitted.' % arg)
            arg_type = type(values[arg])
            try:
                if arg_type == bool:
                    value = (value.lower() == 'true')
                else:
                    value = arg_type(value)
            except:
                raise AttributeError('Could not type value of %s to %s' % (arg, arg_type))
            values[arg] = value

            skip_next = True

        for k, v in values.iteritems():
            setattr(self, k, v)

    def dump(self):
        return ["%s: %s" % (value, parameter) for value, parameter in vars(self).iteritems()]


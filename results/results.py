import ast
import glob
import h5py
import os
import sys

import numpy as np

sys.path.append(os.path.abspath('..'))
from config.config import Config


class Run:

    def __init__(self, path, config):
        self.path = path
        self.name = os.path.split(self.path)[-1]
        self.config = config
        self.load()

    def load(self):
        with open(self.path) as f:
            self.data = []
            for line in f:
                if not line.startswith('#'):
                    datum = ast.literal_eval(line)
                    self.data.append(datum)

    def match(self, datum, **filters):
        return all(
            datum[key] == value
            for key, value
            in filters.items()
        )

    def filter(self, **filters):
        return (
            datum
            for datum in self.data
            if self.match(datum, **filters)
        )

    def history(self, metric, **filters):
        data = self.filter(**filters)
        data = sorted(data, key=lambda d: d['epoch'])
        return (datum[metric] for datum in data)

    def best(self, metric, max_=True, **filters):
        data = self.filter(**filters)
        data = sorted(data, key=lambda d: d[metric])
        result = data[-1] if max_ else data[0]
        return result[metric], result['epoch']

    def metric_eval(self, metric, max_=True, **filters):
        assert('role' not in filters)
        _, epoch = self.best(metric, max_, role='dev', **filters)
        data = list(self.filter(epoch=epoch, role='test', **filters))
        assert(len(data) == 1)
        datum = data[0]
        return datum[metric], datum['epoch']

    def contains(self, task, lang):
        try:
            next(self.filter(task=task, language=lang))
        except StopIteration:
            return False
        return True


config = Config(*sys.argv[1:])

for f in glob.glob(os.path.join(config.log_path, '*', '2_ml*', '*')):
    r = Run(f, config)
    _task = 'dep'
    _lang = 'en'
    _metr = 'gradient_norm'
    if r.contains(_task, _lang):
        print(f, r.history(_metr, task=_task, language=_lang, role="train"))

# r = Run('august_grid/2018-08-09-222952', config)
# print(list(r.history('acc', role='test')))
# print(list(r.best('acc', role='test')))
# print(list(r.best('acc', max_=False, role='test')))
# print(list(r.metric_eval('acc')))
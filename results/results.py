import ast
import os
import sys

from config.config import Config


class Run:

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.load()

    def load(self):
        filename = os.path.join(self.config.log_path, self.name)
        with open(filename) as f:
            self.data = [
                ast.literal_eval(line)
                for line in f
                if not line.startswith('#')
            ]

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




config = Config(*sys.argv[1:])
r = Run('august_grid/2018-08-09-222952', config)
print(list(r.history('acc', role='test')))
print(list(r.best('acc', role='test')))
print(list(r.best('acc', max_=False, role='test')))
print(list(r.metric_eval('acc')))
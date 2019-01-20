import ast
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.abspath('..'))



class Run:

    def __init__(self, path):
        self.path = path
        rest, self.name = os.path.split(self.path)
        rest, self.type = os.path.split(rest)
        _, self.code = os.path.split(rest)
        with open(self.path) as f:
            self.data = ast.literal_eval(f.read())['results']

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

if __name__ == '__main__':
    print(5)

# r = Run('august_grid/2018-08-09-222952', config)
# print(list(r.history('acc', role='test')))
# print(list(r.best('acc', role='test')))
# print(list(r.best('acc', max_=False, role='test')))
# print(list(r.metric_eval('acc')))
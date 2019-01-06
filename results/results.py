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

    def additional_stats(self, datum):
        h5_filepath = os.path.join(
            self.config.model_path,
            f'{self.name}-{datum["epoch"]}-{datum["task"]}-{datum["language"]}-{datum["role"]}.h5')
        output = dict()
        if os.path.isfile(h5_filepath):
            with h5py.File(h5_filepath, 'r') as h5:
                output['gradient_norm'] = h5['gradient_norm'].value

                output['cont_repr_weights_norm'] = np.linalg.norm(h5['cont_repr_weights'])
                output['cont_repr_weights_input_norm'] = np.linalg.norm(h5['cont_repr_weights'], axis=0)
                output['cont_repr_weights_input_norm_avg'] = np.mean(output['cont_repr_weights_input_norm'])
                output['cont_repr_weights_input_norm_std'] = np.std(output['cont_repr_weights_input_norm'])
                output['cont_repr_weights_output_norm'] = np.linalg.norm(h5['cont_repr_weights'], axis=1)
                output['cont_repr_weights_output_norm_avg'] = np.mean(output['cont_repr_weights_output_norm'])
                output['cont_repr_weights_output_norm_std'] = np.mean(output['cont_repr_weights_output_norm'])

                output['cont_repr_weights_grad_norm'] = np.linalg.norm(h5['cont_repr_weights_grad'])
                output['cont_repr_weights_grad_input_norm'] = np.linalg.norm(h5['cont_repr_weights_grad'], axis=0)
                output['cont_repr_weights_grad_input_norm_avg'] = np.mean(output['cont_repr_weights_grad_input_norm'])
                output['cont_repr_weights_grad_input_norm_std'] = np.std(output['cont_repr_weights_grad_input_norm'])
                output['cont_repr_weights_grad_output_norm'] = np.linalg.norm(h5['cont_repr_weights_grad'], axis=1)
                output['cont_repr_weights_grad_output_norm_avg'] = np.mean(output['cont_repr_weights_grad_output_norm'])
                output['cont_repr_weights_grad_output_norm_std'] = np.mean(output['cont_repr_weights_grad_output_norm'])

                output['cont_repr_norm_avg'] = np.mean(np.linalg.norm(h5['cont_repr'], axis=1))
                output['cont_repr_norm_std'] = np.std(np.linalg.norm(h5['cont_repr'], axis=1))


            # cont_repr/cont_repr_grad avg, column avg/std as vector, column avg avg/std, column std avg/std
            print(output)
        return output

    def load(self):
        with open(self.path) as f:
            self.data = []
            for line in f:
                if not line.startswith('#'):
                    datum = ast.literal_eval(line)
                    datum.update(self.additional_stats(datum))
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




config = Config(*sys.argv[1:])

for f in glob.glob(os.path.join(config.log_path, 'gcp', '*')):
    Run(f, config)
# r = Run('august_grid/2018-08-09-222952', config)
# print(list(r.history('acc', role='test')))
# print(list(r.best('acc', role='test')))
# print(list(r.best('acc', max_=False, role='test')))
# print(list(r.metric_eval('acc')))
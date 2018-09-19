import ast
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import itertools


class Experiment:

    def __init__(self, path):
        files = glob.glob('%s*' % path)
        self.runs = [Run(file) for file in files]

    def relevant_runs(self, filters):
        return [run for run in self.runs if run.is_relevant(filters)]

    def graph_results(self, filters, result_filters, metric):

        result_filters.update(filters)

        runs = self.relevant_runs(filters)
        for run in runs:
            results = run.results(metric, result_filters)
            print max(results)
            plt.plot(results, label=run.hyperparameters['tasks'])
        plt.legend()

    def results(self, filters, metric):

        maxs = list()
        dev_filters = dict(filters)
        dev_filters.update({'role': 'dev'})

        test_filters = dict(filters)
        test_filters.update({'role': 'test'})
        for run in self.relevant_runs(filters):
            results = run.results(metric, dev_filters)
            max_id = np.argmax(results)
            results = run.results(metric, test_filters)
            maxs.append(results[max_id])

        return max(maxs)

    def result_history(self, filters, result_filters, metric):

        result_filters.update(filters)
        runs = self.relevant_runs(filters)

        res = []
        for run in runs:
            results = run.results(metric, result_filters)
            res += results
        return res






class Run:

    def __init__(self, path):
        self.records = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    hps = line[3:-1]
                    if re.search('optimizer', hps):
                        self.hyperparameters = ast.literal_eval("{%s}" % hps)
                else:
                    dct = ast.literal_eval(line)
                    dct['file'] = f.name.split('/')[-1]
                    self.records.append(dct)
        self.file = path.split('/')[-1]

    def is_relevant(self, filters):
        return sum([self.is_rec_relevant(filters, record) for record in self.records]) != 0

    def is_rec_relevant(self, filters, rec):
        for f in filters:
            if f not in rec:
                rec[f] = re.search('\'%s\': (.*?),' % f, self.hyperparameters).group(1)
        return sum([rec[key] != filters[key] for key in filters]) == 0

    def results(self, metric, filters):
        return [rec[metric] for rec in self.records if self.is_rec_relevant(filters, rec)]

sizes = ['full', '15000', '5000', '1500', '500', '50']
regimes = ['slst', 'mt', 'ml', 'mlmt']
sizes = ['500']
task = 'dep'
met = 'uas'
for (r, s) in itertools.product(regimes, sizes):
    e = Experiment('/media/piko/Data/fiit/data/cll-para-sharing/logs/august_slst_vs_mlmt/%s/%s/*' % (r,s))
    res = e.results(
        {
            'task': task,
            'language': 'cs'
        },
        met
    )
    hist = e.result_history(
        {
            'task': task,
            'language': 'cs'
        },
        {
            'role': 'dev'
        },
        met
    )
    plt.plot(hist)
    print res, s, r
    # e.graph_results(
    #     {
    #         'task': task,
    #         'language': 'cs'
    #     },
    #     {
    #         'role': 'dev'
    #     },
    #     met
    # )
    # e.graph_results(
    #     {
    #         'task': task,
    #         'language': 'en'
    #     },
    #     {
    #         'role': 'test'
    #     },
    #     met
    # )

plt.show()


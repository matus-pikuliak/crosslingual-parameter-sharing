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
            print results
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
sizes = ['full', '5000', '500']
regimes = ['slst', 'mt', 'ml', 'mlmt']
tasks = [('dep', 'uas'), ('ner', 'f1'), ('pos', 'acc')]

sizes = ['5000']
tasks = [tasks[0]]

fig, ax_lst = plt.subplots(3, 3)  # a figure with a 2x2 grid of Axes
plt.subplots_adjust(hspace=0.5, wspace=0.4)
ax_lst = list(itertools.chain.from_iterable(ax_lst))
for i, (size, (task, metric)) in enumerate(itertools.product(sizes, tasks)):
    ax = ax_lst[i]
    for regime in regimes:
        e = Experiment(
            '/media/fiit/5016BD1B16BD0350/Users/PC/FIIT Google Drive/data/cll-para-sharing/logs/august_slst_vs_mlmt/%s/%s/*' % (
            regime, size))
        hist = e.result_history(
            {
                'task': task,
                'language': 'cs'
            },
            {
                'role': 'dev'
            },
            'loss'
        )
        ax.plot(hist)


    # res = e.results(
    #     {
    #         'task': task,
    #         'language': 'cs'
    #     },
    #     met
    # )
    # hist = e.result_history(
    #     {
    #         'task': task,
    #         'language': 'cs'
    #     },
    #     {
    #         'role': 'dev'
    #     },
    #     met
    # )
    # plt.plot(hist)
    # plt.legend(['slst', 'mt', 'ml', 'mlmt'])
    # print res, s, r
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


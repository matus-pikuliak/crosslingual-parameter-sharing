import ast
import glob
import re
import matplotlib.pyplot as plt


class Experiment:

    def __init__(self, path):
        files = glob.glob('%s*' % path)
        self.runs = [Run(file) for file in files]

    def relevant_runs(self, filters):
        return [run for run in self.runs if run.is_relevant(filters)]

    def graph_results(self, filters, result_filters, metric):

        runs = self.relevant_runs(filters)
        for run in runs:
            results = run.results(metric, result_filters)
            print results
            print max(results)
            plt.plot(results, label='-')
        plt.legend()
        plt.show()


class Run:

    def __init__(self, path):
        self.records = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    hps = line[3:-1]
                    if re.search('optimizer', hps):
                        self.hyperparameters = hps
                else:
                    dct = ast.literal_eval(line)
                    dct['file'] = f.name.split('/')[-1]
                    self.records.append(dct)

    def is_relevant(self, filters):
        return self.is_rec_relevant(filters, self.records[0])

    def is_rec_relevant(self, filters, rec):
        for f in filters:
            if f not in rec:
                rec[f] = re.search('\'%s\': (.*?),' % f, self.hyperparameters).group(1)
        return sum([rec[key] != filters[key] for key in filters]) == 0

    def results(self, metric, filters):
        return [rec[metric] for rec in self.records if self.is_rec_relevant(filters, rec)]




opt = '0.003'
e = Experiment('/media/piko/Data/fiit/data/cll-para-sharing/logs/august_grid/*')
e.graph_results(
    {
        'task': 'pos',
        'learning_rate': opt
    },
    {
        'role': 'dev'
    },
    'acc'
)

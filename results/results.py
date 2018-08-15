import ast
import glob
import re
import matplotlib.pyplot as plt


class Experiment:

    def __init__(self, path):
        files = glob.glob('%s*' % path)
        self.runs = [Run(file) for file in files]

    def results(self, filters, result_filters, metric):
        relevant_runs = [run for run in self.runs if run.is_relevant(filters)]
        return [run.results(metric, result_filters) for run in relevant_runs]

    def graph_results(self, *args):

        results = self.results(*args)
        for rec in results:
            print max(rec)
            plt.plot(rec)#, label='%s' % (run.lang))
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
                    if re.search('optimizer: (.*?),', hps):
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
                rec[f] = re.search('%s: (.*?),' % f, self.hyperparameters).group(1)
        return sum([rec[key] != filters[key] for key in filters]) == 0

    def results(self, metric, filters):
        if metric == 'f1':
            def f1(p, r):
                return 2*p*r/(p+r+ 1e-12)

            return [f1(rec['precision'], rec['recall']) for rec in self.records if self.is_rec_relevant(filters, rec)]
        else:
            return [rec[metric] for rec in self.records if self.is_rec_relevant(filters, rec)]




opt = '0.003'
print opt
e = Experiment('/media/piko/Data/fiit/data/cll-para-sharing/logs/august_grid/*')
e.graph_results(
    {
        'task': 'dep',
        'learning_rate': opt
    },
    {
        'role': 'dev'
    },
    'las'
)

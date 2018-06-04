import paths

class Run:

    runs = []

    @classmethod
    def get_runs(cls, lang=None, task=None, role=None, run=None, file=None):
        relevant_runs = []
        for r in cls.runs:
            if (
            (r.lang == lang or lang is None) and
            (r.task == task or task is None) and
            (r.role == role or role is None) and
            (r.run == run or run is None) and
            (r.file == file or file is None)):
                relevant_runs.append(r)
        return relevant_runs

    @classmethod
    def process_epoch(cls, epoch):
        run = cls.get_runs(epoch['language'], epoch['task'], epoch['role'], epoch['run'], epoch['file'])
        if not run:
            run = [Run(epoch)]
        run[0].add_epoch(epoch)

    def __init__(self, epoch):
        self.lang = epoch['language']
        self.task = epoch['task']
        self.role = epoch['role']
        self.run = epoch['run']
        self.file = epoch['file']
        self.results = {}
        self.max_epoch = 0
        self.__class__.runs.append(self)

    def add_epoch(self, epoch):
        epoch_id = epoch['epoch']
        self.max_epoch = max(self.max_epoch, epoch_id)
        self.results[epoch_id] = epoch

    def read_metric(self, metric_name):
        buffer = []
        for e in xrange(self.max_epoch):
            epoch_id = e + 1
            buffer.append(self.results[epoch_id][metric_name])
        return buffer

    def max_metric(self, metric_name):
        buff = self.read_metric(metric_name)
        return max(buff)

    def max_metric_epoch(self, metric_name):
        buff = self.read_metric(metric_name)
        max = 0
        max_id = 0
        for i, val in enumerate(buff):
            if val > max:
                max = val
                max_id = i+1
        return max_id

def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def is_float(str):
    try:
        float(str)
        return not is_int(str)
    except ValueError:
        return False


def is_str(str):
    return not (is_float(str) or is_int(str))

import glob
files = glob.glob(paths.paths['log_path']+'/ner_es_lr/*')
records = []
for file in files:
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                dct = dict([i.split(': ') for i in line.split(', ')])
                for key in dct:
                    if is_int(dct[key]): dct[key] = int(dct[key])
                    if is_float(dct[key]): dct[key] = float(dct[key])
                dct['file'] = f
                Run.process_epoch(dct)

import matplotlib.pyplot as plt

en_pos = Run.get_runs(role='dev')
for run in en_pos:
    print run.max_metric('f1'), run.run
    plt.plot(run.read_metric('f1'), label=run.file)
plt.legend()
plt.show()

import ast
import os


class Run:

    def eager(func):
        def wrap(self, *args, **kwargs):
            self.load()
            return func(self, *args, **kwargs)
        return wrap

    def __init__(self, path, type, name, server, autoload=True):
        """
        :param path:
        :param type: stsl / mt / ml / mtml
        :param code: settings that were used
        """
        self.path = path
        rest, self.filename = os.path.split(self.path)
        self.type = type
        self.name = name
        self.server = server
        self.loaded = False
        if autoload:
            self.load()

    def load(self):
        if not self.loaded:
            with open(self.path) as f:
                log = ast.literal_eval(f.read())
                self.data = log['results']
                self.config = log['config']
            self.loaded = True

    def match_datum(self, datum, **filters):
        return all(
            datum[key] == value
            for key, value
            in filters.items()
        )

    def match(self, type=None, name=None, contains=(), **config):
        fast_check = (
            (name is None or name == self.name) and
            (type is None or type == self.type)
        )

        if not fast_check:
            return False

        self.load()
        return (
            all(self.contains(*tl) for tl in contains) and
            all(self.config[key] == value for key, value in config.items())
        )

    def filter_data(self, **filters):
        return (
            datum
            for datum in self.data
            if self.match_datum(datum, **filters)
        )

    @eager
    def history(self, metric=None, **filters):
        metric, maximize = self.set_metric_and_meximize(metric, None, **filters)
        data = self.filter_data(**filters)
        data = sorted(data, key=lambda d: d['epoch'])
        return (datum[metric] for datum in data)

    @eager
    def best(self, metric=None, maximize=None, **filters):
        metric, maximize = self.set_metric_and_meximize(metric, maximize, **filters)
        data = self.filter_data(**filters)
        data = sorted(data, key=lambda d: d[metric])
        result = data[-1] if maximize else data[0]
        return result[metric], result['epoch']

    @eager
    def metric_eval(self, metric=None, maximize=None, **filters):
        metric, maximize = self.set_metric_and_meximize(metric, maximize, **filters)
        assert('role' not in filters)
        _, epoch = self.best(metric, maximize, role='dev', **filters)
        data = list(self.filter_data(epoch=epoch, role='test', **filters))
        assert(len(data) == 1)
        datum = data[0]
        return datum[metric], epoch

    @eager
    def contains(self, task, lang):
        try:
            next(self.filter_data(task=task, language=lang))
            return True
        except StopIteration:
            return False

    @staticmethod
    def default_maximize(metric):
        return {
            'las': True,
            'uas': True,
            'perplexity': False,
            'chunk_f1': True,
            'acc': True,
            'loss': False,
            'adv_loss': False,
        }[metric]

    @staticmethod
    def default_metric(task):
        return {
            'dep': 'las',
            'lmo': 'perplexity',
            'ner': 'chunk_f1',
            'pos': 'acc'
        }[task]

    def set_metric_and_meximize(self, metric, maximize, **filters):
        if metric is None:
            metric = self.default_metric(filters['task'])
        if maximize is None:
            maximize = self.default_maximize(metric)
        return metric, maximize

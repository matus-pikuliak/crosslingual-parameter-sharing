import os

from data2.word_normalization import word_normalization


class Dataset:

    def __new__(cls, task, *args):
        '''
        Dark magic that overwrites __new__ method for subclasses so they do not call this function by inheritance.
        Subclasses must be listed here to avoid circular dependencies.
        The whole point of this exercise is so we can create new datasets by calling Dataset(*args) and it will
        automatically create suitable Dataset subclass.
        '''

        tasks_model = (
            (('pos', 'ner'), 'sqt'),
            (('lmo'), 'lmo'),
            (('nli'), 'nli'),
            (('dep'), 'dep')
        )

        def module_name(model):
            return f'data2.dataset_{model}'

        def subclass_name(model):
            return f'{model.upper()}Dataset'

        # loads relevant subclasses defined in tassk_models
        datasets = {model: __import__(module_name(model), fromlist=[subclass_name(model)])
                    for _, model in tasks_model}

        # overwrites subclasses' __new__ method
        def subclass_new(cls, *_, **__):
            return object.__new__(cls)

        for subclass in Dataset.__subclasses__():
            subclass.__new__ = subclass_new

        # creates suitable Dataset subclass based on the task user specified
        for tasks, model in tasks_model:
            if task in tasks:
                subclass = getattr(datasets[model], subclass_name(model))
                return subclass(task, *args)

        raise AttributeError(f'Unknown task: {task}')

    def __init__(self, task, lang, role, config, data_loader):
        self.task = task
        self.lang = lang
        self.role = role
        self.config = config
        self.filename = os.path.join(config.data_path, task, lang, role)
        self.dl = data_loader

    def __str__(self):
        return f'Dataset {self.task} {self.lang} {self.role} ({self.__class__.__name__})'

    def word_to_id(self, word):
        return self.dl.lang_vocabs[self.lang].word_to_id(word)

    def label_to_id(self, label):
        return self.dl.task_vocabs[self.task].label_to_id(label)

    def word_to_char_ids(self, word):
        return self.dl.char_vocab.word_to_ids(word)

    def load(self):
        raise NotImplementedError

    def load_hists(self):
        self.lang_hist = {}
        self.char_hist = {}
        self.task_hist = {}

        with open(self.filename, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) > 0:
                    word = tokens[0]
                    self.lang_hist.setdefault(word, 0)
                    self.lang_hist[word] += 1
                    for char in word:
                        self.char_hist.setdefault(char, 0)
                        self.char_hist[char] += 1

                if len(tokens) > 1:
                    label = tokens[1]
                    self.task_hist.setdefault(label, 0)
                    self.task_hist[label] += 1

    def del_hists(self):
        self.lang_hist.clear()
        self.task_hist.clear()
        self.char_hist.clear()

    def get_hist(self, hist_type):
        try:
            hist = {
                'lang': self.lang_hist,
                'char': self.char_hist,
                'task': self.task_hist
            }
        except AttributeError:
            raise RuntimeError(f'Dataset.load_hists() was not called yet: ({self})')
        return hist[hist_type]

    def read_raw_samples(self):
        with open(self.filename, 'r') as f:
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line.split())
                else:
                    yield lines
                    lines = []

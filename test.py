# -*- coding: utf-8 -*-

import sys

from config.config import Config
from data.data_manager import DataManager
from model.model import Model

config = Config(sys.argv[1:])

a = [
    [('ner', 'en')],
    [('ner', 'es')],
    [('ner', 'de')],
    [('ner', 'cs')],
    [('pos', 'en')],
    [('pos', 'es')],
    [('pos', 'de')],
    [('pos', 'cs')],
    ]
b = [
    [
        ('ner', 'en'),
        ('pos', 'en')
    ],
    [
        ('ner', 'es'),
        ('pos', 'es')
    ],
    [
        ('ner', 'de'),
        ('pos', 'de')
    ],
    [
        ('ner', 'cs'),
        ('pos', 'cs')
    ]
    ]
c = [
    [
        ('pos', 'en'),
        ('pos', 'es'),
        ('pos', 'de'),
        ('pos', 'cs'),
    ],
    [
        ('ner', 'en'),
        ('ner', 'es'),
        ('ner', 'de'),
        ('ner', 'cs'),
    ]
    ]
d = [
    [
        ('pos', 'en'),
        ('pos', 'es'),
        ('pos', 'de'),
        ('pos', 'cs'),
        ('ner', 'en'),
        ('ner', 'es'),
        ('ner', 'de'),
        ('ner', 'cs'),
    ]
    ]

if config.tasks is None:
    train_sets = d
    tls = [(task, lang) for task in ['ner', 'pos'] for lang in ['en', 'es', 'de', 'cs']]
    # train_sets = [[('pos','en'), ('pos','es'), ('ner','en'), ('ner', 'es')]]
    # tls = [(task, lang) for task in ['ner', 'pos'] for lang in ['en', 'es']]
else:
    train_sets = [config.tasks] # [[t] for t in config.tasks]
    tls = config.tasks

dm = DataManager(tls=tls, config=config)
dm.print_stats()
exit()
dm.prepare()

for train_set in train_sets:
    model = Model(dm, config)
    model.build_graph()
    model.run_experiment(
        train=train_set,
        test=[],
        epochs=config.epochs
    )
    model.close()

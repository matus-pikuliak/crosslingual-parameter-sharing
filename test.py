# -*- coding: utf-8 -*-

import os
import sys
import time

from config.config import Config
from data.data_manager import DataManager
from logs.logger import Logger
from model.model import Model

config = Config(sys.argv[1:])
config.initialize_logger()  # Logger can now be used

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
    train_sets = a+b+c+d
    tls = [(task, lang) for task in ['ner', 'pos'] for lang in ['en', 'es', 'de', 'cs']]
else:
    train_sets = config.tasks
    tls = config.tasks

dm = DataManager(tls=tls, config=config)
dm.prepare()

for train_set in train_sets:
    Logger().log_critical('Run started.')
    logger = Logger(config.log_path, time.strftime('%Y-%m-%d-%H%M%S', time.gmtime()))
    model = Model(dm, config, Logger())
    model.build_graph()
    model.run_experiment(
        train=train_set,
        test=[],
        epochs=config.epochs
    )
    model.close()
    Logger().log_critical('Run done.')

os.system('notify-send "SUCCESS" "well done beb"')

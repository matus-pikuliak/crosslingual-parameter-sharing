# -*- coding: utf-8 -*-

from data.data_manager import DataManager
from config import Config
from model.model import Model
import os
import sys
import time
from logs.logger import Logger

config = Config(sys.argv[1:])
config.initialize_logger() # Logger can now be used
dm = DataManager(tasks=['pos', 'ner'], languages=['en'], config=config)
dm.prepare()

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

train_sets = a+b+c+d

if config.setup == 'posen':
    train_sets = [[('pos', 'en')]]

if config.setup == 'neres':
    train_sets = [[('ner', 'es')]]

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

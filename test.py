# -*- coding: utf-8 -*-

from data.cache import Cache
from config import Config
from model.model import Model
import os
import sys
from logs.logger import Logger
import time

config = Config(sys.argv[1:])
cache = Cache(config)
cache = cache.load()

train_sets = [
    [('ner', 'en')],
    [('ner', 'es')],
    [('ner', 'de')],
    [('ner', 'cs')],
    [('pos', 'en')],
    [('pos', 'es')],
    [('pos', 'de')],
    [('pos', 'cs')],
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
    ],
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
    ],
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

for train_set in train_sets:
    logger = Logger(config.log_path, time.strftime('%Y-%m-%d-%H%M%S', time.gmtime()))
    model = Model(cache, config, logger)
    model.build_graph()
    model.run_experiment(
        train=train_set,
        test=[],
        epochs=config.epochs
    )
    model.close()

os.system('notify-send "SUCCESS" "well done beb"')

# TODO:
#       - check engglish NER datasets
#       - vytvor small testing set
#       - saving
#       - regularization
#       - v cache.py pouzivam lower(). Treba vlastne embeddings + konvo?
#       - dependency parsing
#       - machine translation
#       - language modeling

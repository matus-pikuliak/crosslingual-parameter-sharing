# -*- coding: utf-8 -*-

import os
import time
import sys
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
#       - vytvor small testing set
#       - learning rate decay
#       - saving
#       - regularization
#       - v cache.py pouzivam lower(). Treba vlastne embeddings + konvo?
#       - dependency parsing
#       - machine translation
#       - language modeling

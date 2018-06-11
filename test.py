# -*- coding: utf-8 -*-

from data.cache import Cache
from config import Config
from model.model import Model
import os
import sys
from logs.logger import Logger
import time

from logs.slack_notifier import SlackNotifier
from private import slack_config
slack_notifier = SlackNotifier(slack_config['token'], slack_config['channel'])

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

if config.setup == 'default':
    train_sets = a+b+c+d

if config.setup == 'posen':
    train_sets = [[('pos', 'en')]]

if config.setup == 'neres':
    train_sets = [[('ner', 'es')]]

for train_set in train_sets:
    slack_notifier.send('Run started.')
    logger = Logger(config.log_path, time.strftime('%Y-%m-%d-%H%M%S', time.gmtime()))
    model = Model(cache, config, logger)
    model.build_graph()
    model.run_experiment(
        train=train_set,
        test=[],
        epochs=config.epochs
    )
    model.close()
    slack_notifier.send('Run done.')

os.system('notify-send "SUCCESS" "well done beb"')

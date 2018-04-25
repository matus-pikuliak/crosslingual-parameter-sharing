# -*- coding: utf-8 -*-

from data.cache import Cache
from config import Config
from model.model import Model
import os

config = Config()
cache = Cache(config)
cache = cache.load()

# dt = cache.fetch_dataset('ner', 'de', 'train')
# unk = cache.lang_dicts['de'][1]['<unk>']
# print unk
# zer = cache.task_dicts['ner'][1]['O']
# all = 0
# all_u = 0
# ner = 0
# ner_u = 0
# for s in dt.samples:
#     for i in xrange(len(s[0])):
#         all += 1
#         if s[0][i] == unk:
#             all_u += 1
#         if s[1][i] != zer:
#             ner += 1
#             if s[0][i] == unk:
#                 ner_u += 1
# print all, all_u, ner, ner_u
#
#
#
# exit()

def run_test(train, cache, config, epoch):
    name = ' '.join([' '.join(t) for t in train])
    with open('./logs.txt', 'a') as f:
        f.write('Now training '+name+'\n')
    model = Model(cache, config)
    model.build_graph()
    for i in xrange(epoch):
        model.run_epoch(i,
            train=train,
            test=[]
        )
    model.close()

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
    run_test(train_set, cache, config, 30)

os.system('notify-send "SUCCESS" "well done beb"')

# TODO:
#       - check NER datasets
#       - vytvor small testing set
#       - summaries
#       - regularization
#       - logging
#       - v cache.py pouzivam lower(). Treba vlastne embeddings
#       - model saving
#       - dependency parsing
#       - machine translation
#       - language modeling

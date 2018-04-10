from data.cache import Cache
from config import Config
from model.model import Model
import os

config = Config()
cache = Cache(config)
cache = cache.load()

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
    run_test(train_set, cache, config, 15)

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

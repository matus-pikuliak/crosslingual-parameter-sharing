from data.cache import Cache
from config import Config
from model.model import Model

import os

config = Config()
cache = Cache(config)
cache = cache.load()

dt = cache.fetch_dataset('pos', 'cs', 'train')
print len(dt)
exit()

model = Model(cache, config)
model.build_graph()
for i in xrange(10):
    model.run_epoch(i,
        train=[
            ('ner', 'en'),
            ('ner', 'de')
        ],
        test=[
            #('ner', 'en')
        ]
    )
os.system('notify-send "SUCCESS" "well done beb"')

# TODO:
#       - check NER datasets
#       - summaries
#       - regularization
#       - logging
#       - model saving
#       - dependency parsing
#       - machine translation
#       - language modeling

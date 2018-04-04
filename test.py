from data.cache import Cache
from config import Config
from model.model import Model
import pickle
import os


config = Config()
cache = Cache(config)
cache.create(languages=['en'], tasks=['ner'])

model = Model(cache, config)
model.build_graph()
for i in xrange(1):
    model.run_epoch(
        cache.fetch_dataset(language='en', task='ner', role='train'),
        cache.fetch_dataset(language='en', task='ner', role='dev'),
    )
os.system('notify-send "SUCCESS" "well done beb"')

# todo: model saving
#       summaries
#       logging
#       cache saving

from data.cache import Cache
from config import Config
from model.model import Model
import pickle
import os

try:
    config = Config()
    cache = Cache(config)
    cache.create(languages=['cs','en','de'])
    model = Model(cache, config)
    model.build_graph()
    for i in xrange(10):
        model.run_epoch(cache.datasets.train, cache.datasets.dev)
    os.system('notify-send "SUCCESS" "well done beb"')
except:
    os.system('notify-send "FAIL" "oh no"')

# todo: model saving
#       summaries
#       logging
#
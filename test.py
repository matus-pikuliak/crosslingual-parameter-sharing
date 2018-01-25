from data.cache import Cache
from config import Config
import pickle

config = Config()
cache = Cache(config)
cache.create()

from data.cache import Cache
from config import Config

config = Config()
cache = Cache(config)
cache.create()

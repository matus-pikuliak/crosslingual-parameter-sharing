from config import Config
from data.cache import Cache

config = Config()
cache = Cache(config)
cache.create(languages=['en', 'de', 'es', 'cs'], tasks=['pos', 'ner'])
cache.save()
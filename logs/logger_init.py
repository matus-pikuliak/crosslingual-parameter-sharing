from .logger import Logger
from .default_logger import DefaultLogger
from .debug_logger import DebugLogger
from .production_logger import ProductionLogger


class LoggerInit:

    def __init__(self, type, *args, **kwargs):
        logger_type = {
            'default': DefaultLogger,
            'debug': DebugLogger,
            'production': ProductionLogger
        }[type]
        self.logger = logger_type(*args, **kwargs)

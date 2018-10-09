from .logger import Logger


class ProductionLogger(Logger):

    def log_debug(self, msg):
        None

    def log_error(self, msg):
        self.stderr(msg)

    def log_message(self, msg):
        self.file('# %s'%msg)

    def log_result(self, msg):
        self.file(msg)

    def log_critical(self, msg):
        self.slack(msg)
        self.file('# %s' % msg)
from logger import Logger
import os


class DefaultLogger(Logger):

    def log_debug(self, msg):
        self.stdout(msg)

    def log_error(self, msg):
        self.stderr(msg)

    def log_message(self, msg):
        self.file('# %s'%msg)

    def log_result(self, msg):
        self.file(msg)

    def log_critical(self, msg):
        self.file('# %s' % msg)
        os.system('notify-send %s' % msg)

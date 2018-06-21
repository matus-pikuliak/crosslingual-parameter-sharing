from logger import Logger


class DebugLogger(Logger):

    def log_debug(self, msg):
        self.stdout(msg)

    def log_error(self, msg):
        self.stderr(msg)

    def log_message(self, msg):
        self.stdout(msg)

    def log_result(self, msg):
        self.stdout(msg)

    def log_critical(self, msg):
        self.stdout(msg)
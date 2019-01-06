from logs.logger import Logger


class DefaultLogger(Logger):

    def log_message(self, msg):
        self.stdout(msg)

    def log_result(self, msg):
        self.stdout(msg)
        self.file(msg)

    def log_critical(self, msg):
        self.stdout(msg)
        self.file(f'# {msg}')
        self.system(msg)


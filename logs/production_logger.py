from logs.logger import Logger


class ProductionLogger(Logger):

    def log_message(self, msg):
        self.stdout(msg)
        self.file(f'# {msg}')

    def log_result(self, msg):
        self.stdout(msg)
        self.file(msg)

    def log_critical(self, msg):
        self.stdout(msg)
        self.slack(f'{self.server_name}: {msg}')
        self.file(f'# {msg}')

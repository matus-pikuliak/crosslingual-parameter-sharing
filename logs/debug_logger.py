from logs.logger import Logger


class DebugLogger(Logger):

    def log_message(self, msg):
        self.stdout(msg)

    def log_result(self, msg):
        for k, v in msg.items():
            message = f'{k}: {v}'
            self.stdout(message)

    def log_critical(self, msg):
        self.stdout(msg)

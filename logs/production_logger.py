from logs.logger import Logger


class ProductionLogger(Logger):

    def __init__(self, **kwargs):
        Logger.__init__(self, **kwargs)
        self.results = dict()

    def log_message(self, msg):
        self.stdout(msg)

    def log_result(self, msg):
        for k, v in msg.items():
            self.stdout(f'{k}: {v}')
            if k in self.results:
                try:
                    self.results[k].append(v)
                except AttributeError:
                    self.results[k] = [self.results[k]] + [v]
            else:
                self.results[k] = v
        self.file(self.results, mode='w')

    def log_critical(self, msg):
        self.stdout(msg)
        self.slack(f'{self.server_name}: {msg}')

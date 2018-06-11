class Logger:

    def __init__(self, dir, name):
        self.filename = dir + name

    def log_m(self, message):
        if isinstance(message, list):
            for msg in message:
                self.log_m(msg)
        else:
            with open(self.filename, 'a') as f:
                f.write('# '+ message + '\n')

    def log_r(self, result):
        with open(self.filename, 'a') as f:
            f.write(', '.join(['%s: %s' % (k, result[k]) for k in result]) + '\n')

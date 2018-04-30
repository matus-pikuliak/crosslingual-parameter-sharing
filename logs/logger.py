class Logger:

    def __init__(self, dir, name):
        self.filename = dir + name

    def log_m(self, message):
        with open(self.filename, 'a') as f:
            f.write('# '+ message + '\n')

    def log_r(self, result):
        with open(self.filename, 'a') as f:
            f.write(', '.join(['%s: %s' % (k, result[k]) for k in result]) + '\n')

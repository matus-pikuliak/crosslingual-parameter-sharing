from private import paths


class Config:

    def __init__(self, parameters=[]):
        for path in paths:
            setattr(self, path, paths[path])

        self.setup = 'default'

        self.word_emb_type = 'static' # static/fasttext
        self.word_emb_size = 300
        self.train_embeddings = False

        self.crf_sharing = False # Share CRF across languages?
        self.lstm_size = 300

        self.learning_rate = 0.003
        self.batch_size = 32
        self.epoch_steps = 512
        self.epochs = 60
        self.clip = 1
        self.dropout = 0.5
        self.optimizer = 'rmsprop' # rmsprop/adagrad/adam/sgd

        # settings
        self.use_gpu = True

        for i in xrange(0, len(parameters)-1, 2):
            parameter = parameters[i]
            value = parameters[i+1]
            try:
                if isinstance(getattr(self, parameter), bool):
                    setattr(self, parameter, True if value == 'True' else False)
                elif isinstance(getattr(self, parameter), str):
                    setattr(self, parameter, value)
                elif isinstance(getattr(self, parameter), int):
                    setattr(self, parameter, int(value))
                elif isinstance(getattr(self, parameter), float):
                    setattr(self, parameter, float(value))
            except AttributeError:
                print "Wrong parameter provided: "+parameter


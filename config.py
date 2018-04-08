from paths import paths


class Config:

    def __init__(self, parameters=[]):
        for path in paths:
            setattr(self, path, paths[path])

        # hyperparameters
        self.word_emb_type = 'static' # static/fasttext
        self.word_emb_size = 300
        self.train_embeddings = False

        self.lstm_size = 100
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.epoch_steps = 256

        # settings
        self.clip = 1

        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])



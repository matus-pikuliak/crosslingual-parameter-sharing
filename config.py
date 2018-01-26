from paths import paths


class Config:

    def __init__(self, parameters=[]):
        for path in paths:
            setattr(self, path, paths[path])

        self.word_emb_type = 'static' # static/fasttext
        self.word_emb_size = 300
        self.train_embeddings = True

        self.lstm_size = 100
        self.proj_size = 100

        self.learning_rate = 1e-3

        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])



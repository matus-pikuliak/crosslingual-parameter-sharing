
class Dataset():

    def __init__(self, language, task, role, sentences):
        self.language = language
        self.task = task
        self.role = role
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def get_sample(self):
        return self.sentences[0]

    def get_batch(self, batch_size):
        return self.sentences[0:50]
        # zero padding

    def reset(self):
        counter = 0


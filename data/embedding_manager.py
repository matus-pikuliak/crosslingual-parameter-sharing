import os
import codecs
import numpy as np
import sys


class EmbeddingManager:

    def __init__(self, languages, config):
        self.languages = languages
        self.config = config
        self.embeddings = dict()

    def load_files(self):
        for lang in self.languages:
            self.embeddings[lang] = dict()
            emb_file = os.path.join(self.config.emb_path, lang)
            with codecs.open(emb_file, encoding='utf-8') as f:
                f.readline()  # first init line in emb files
                for line in f:
                    try:
                        word, vector = line.split(' ', 1)
                        self.embeddings[lang][word] = np.array([float(val) for val in vector.split(' ')])
                    except:
                        if self.config.setup != 'production':
                            print(("Warning: there is a ill formatted line for language %s: '%s'" % (lang, line)).
                                  encode(sys.stdout.encoding, 'ignore'))

    def vocab(self, lang):
        return set(self.embeddings[lang].keys())

    def embedding(self, lang, word):
        return self.embeddings[lang][word]

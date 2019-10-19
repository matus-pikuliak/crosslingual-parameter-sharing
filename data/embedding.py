import os

import numpy as np
import tensorflow as tf

from utils.general_utils import split_iter


class Embeddings:

    def __init__(self, lang, config):
        self.config = config
        self.lang = lang

    def matrix(self, vocab):
        print(f'Loading {self.lang} embedding matrix.')
        emb_matrix = np.zeros(
            shape=(len(vocab), self.config.word_emb_size),
            dtype=np.float)

        # used for word_emb_type 'mwe_projected'
        order = np.random.permutation(self.config.word_emb_size)

        for line in self.read_lines():
            line_iter = split_iter(line)
            word = next(line_iter)

            if word in vocab:
                id = vocab.word_to_id(word)
                vec = np.array([float(n) for n in line_iter])

                if self.config.word_emb_type == 'mwe':
                    emb_matrix[id] = vec
                elif self.config.word_emb_type == 'random':
                    vec = np.random.random(self.config.word_emb_size)
                    norm = np.linalg.norm(vec)
                    emb_matrix[id] = vec / norm
                elif self.config.word_emb_type == 'mwe_rotated':
                    emb_matrix[id] = vec[order]
                else:
                    raise RuntimeError('Wrong embedding types(must be "mwe", "random" or "mwe_projected").')

        with tf.variable_scope('word_embeddings', reuse=tf.AUTO_REUSE):
            emb_matrix = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.cast(
                    x=emb_matrix,
                    dtype=tf.float32),
                trainable=self.config.train_emb,
                name=f'word_embedding_matrix_{self.lang}')
        print('Loaded.')
        return emb_matrix

    def vocab(self):
        return {
            next(split_iter(line)): 0
            for line
            in self.read_lines()
        }

    def read_lines(self):
        filename = os.path.join(self.config.emb_path, self.lang)
        with open(filename) as f:
            next(f)
            for line in f:
                line_iter = split_iter(line)
                _, first = next(line_iter), next(line_iter)
                try:
                    float(first)  # Gets rid of words with whitespaces in them
                    yield line
                except ValueError:
                    pass

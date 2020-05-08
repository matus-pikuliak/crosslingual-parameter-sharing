"""
python get_representations.py log_path model_path task-lang1 task-lang2 ...
"""
import os
import sys

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config.config import Config
from data.data_loader import DataLoader
from model.orchestrator import Orchestrator


def get_desired(orch, *tls):
    desired = []
    for tl in tls:
        task, lang = tl
        model = orch.models[(task, lang)]
        test_set = model.create_sets(is_train=False, role='test', task=task, lang=lang)
        iterator = test_set[0].iterator
        for _ in range(5):
            _, sentence_lengths, _, _, label_ids = next(iterator)
            for length, labels in zip(sentence_lengths, label_ids):
                desired.extend(labels[:length])
    return np.array(desired)


def reduce_dim(x, pca=True, tsne=True):
    print('Reducing dimensionality.')
    if tsne and pca:
        x = PCA(n_components=50).fit_transform(x)
        x = TSNE(n_components=2, n_iter=1000).fit_transform(x)
    elif tsne:
        x = TSNE(n_components=2, n_iter=500).fit_transform(x)
    elif pca:
        x = PCA(n_components=2).fit_transform(x)
    print('Done.')
    return x


def show_representations(log_path, model_path, lstm, tls):

    config = Config.load_from_log(log_path)
    config.values['use_gpu'] = False
    dl = DataLoader(config)
    dl.load()

    with Orchestrator(dl, config) as orch:
        orch.load(model_path)
        representations = orch.get_representations(
            batch_count=5,
            lstm=lstm,
            tls=tls)
        print('Representations obtained.')

        data = np.vstack([r for r in representations.values()])
        data = reduce_dim(data)
        x, y = data[:, 0], data[:, 1]
        print('Dimensionality reduced')

        legend = dict()
        for i, (key, value) in enumerate(representations.items()):
            print(i, key)
            legend[i] = value

        c1 = get_desired(orch, *tls)
        c2 = np.concatenate(
            [np.array([i] * len(r))
             for i, r
             in legend.items()
             ])

    return x, y, c1, c2


if __name__ == '__main__':
    show_representations(*sys.argv[1:])

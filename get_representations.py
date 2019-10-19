"""
python get_representations.py log_path model_path task-lang1 task-lang2 ...
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tkinter
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import colors as mcolors
plt_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

from config.config import Config
from data.data_loader import DataLoader
from model.orchestrator import Orchestrator


def get_representations(log_path, model_path, *tls):
    config = Config.load_from_log(log_path)
    dl = DataLoader(config)
    dl.load()

    with Orchestrator(dl, config) as orch:
        orch.load(model_path)
        return orch.get_representations(
            batch_count=5,
            tls=tls)


def get_desired(log_path, *tls):
    config = Config.load_from_log(log_path)
    dl = DataLoader(config)
    dl.load()

    with Orchestrator(dl, config) as orch:
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
        x = TSNE(n_components=2, n_iter=1000).fit_transform(x)
    elif pca:
        x = PCA(n_components=2).fit_transform(x)
    print('Done.')
    return x


def show_representations(log_path, model_path, *tls):
    representations = get_representations(log_path, model_path, *tls)
    data = np.vstack([r for r in representations.values()])
    data = reduce_dim(data)
    x, y = data[:, 0], data[:, 1]


    # scheme = ['gold', 'red', 'deepskyblue', 'green']


    fig, axes = plt.subplots(1, 2, figsize=(5, 7), squeeze=True)
    scheme = ['gold', 'red', 'deepskyblue', 'green']

    c = get_desired(log_path, *tls)
    ax = axes[0]
    ax.scatter(x, y, c=c, s=2)

    c = np.concatenate(
        [np.array([i] * len(r))
         for i, r
         in enumerate(representations.values())
        ])

    ax = axes[1]
    ax.scatter(x, y, c=c, s=2)

    # legend_elements = [
    #     Patch(color=scheme[i], label=lang)
    #     for i, lang in enumerate(languages)]
    # ax.legend(handles=legend_elements)

    _, name = os.path.split(log_path)
    plt.show()
    # plt.savefig('/home/fiit/logs/images/'+name)


if __name__ == '__main__':
    show_representations(*sys.argv[1:])

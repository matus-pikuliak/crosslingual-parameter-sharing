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
        return orch.get_representations([tuple(tl.split('-')) for tl in tls])


def get_scatter_data(representations, pca=True, tsne=True):
    x = np.vstack([r for r in representations.values()])

    if tsne and pca:
        x = PCA(n_components=50).fit_transform(x)
        x = TSNE(n_components=2, n_iter=1000).fit_transform(x)
    elif tsne:
        x = TSNE(n_components=2, n_iter=1000).fit_transform(x)
    elif pca:
        x = PCA(n_components=2).fit_transform(x)

    scheme = ['gold', 'red', 'deepskyblue', 'green']
    new_x = np.zeros((x.shape[0], x.shape[1] + 1))
    new_x[:, :-1] = x
    x = new_x
    i = 0
    for rep_id, rep in enumerate(representations.values()):
        for _ in rep:
            x[i][2] = rep_id
            i += 1

    np.random.shuffle(x)
    x = x[:2000, ...]
    x, y, c = zip(*x)
    c = np.rint(c)
    c = [plt_colors[scheme[int(i)]] for i in c]
    return x, y, c, representations.keys()


def show_representations(log_path, *args):
    representations = get_representations(log_path, *args)
    x, y, c, languages = get_scatter_data(representations)

    fig, axes = plt.subplots(1, 1, figsize=(5, 7), squeeze=False)
    scheme = ['gold', 'red', 'deepskyblue', 'green']

    ax = axes[0][0]
    ax.scatter(x, y, c=c, s=2)
    legend_elements = [
        Patch(color=scheme[i], label=lang)
        for i, lang in enumerate(languages)]
    ax.legend(handles=legend_elements)

    _, name = os.path.split(log_path)
    plt.savefig('/home/mpikuliak/logs/images/'+name)


if __name__ == '__main__':
    show_representations(*sys.argv[1:])

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import colors as mcolors
plt_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

sys.path.append(os.path.abspath('..'))
from get_representations import get_representations

tsne = True
pca = True


def get_scatter_data(run_name):
    repr = get_representations(run_name)
    # repr = {i: np.random.random((200,200) )for i in range(4)}
    x = np.vstack([r for r in repr.values()])

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
    for rep_id, rep in enumerate(repr.values()):
        for _ in rep:
            x[i][2] = rep_id
            i += 1

    np.random.shuffle(x)
    x, y, c = zip(*x)
    c = np.rint(c)
    c = [plt_colors[scheme[int(i)]] for i in c]
    return x, y, c, repr.keys()


run_names = sys.argv[1:]
fig, axes = plt.subplots(len(run_names), 1, figsize=(5, 7), squeeze=False)

scheme = ['gold', 'red', 'deepskyblue', 'green']

for i, run_name in enumerate(run_names):
    ax = axes[i][0]
    x, y, c, languages = get_scatter_data(run_name)
    ax.scatter(x, y, c=c, s=2)
    legend_elements = [
        Patch(color=scheme[i], label=lang)
        for i, lang in enumerate(languages)]

    if i == len(run_names) - 1:
        ax.legend(handles=legend_elements)

plt.show()
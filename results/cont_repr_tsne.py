import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath('..'))
from get_representations import get_representations

tsne = True
pca = True

repr = get_representations(sys.argv[1])
x = np.vstack([r for r in repr.values()])

print('Representations loaded.')
print(x.shape)

if tsne and pca:
    x = PCA(n_components=50).fit_transform(x)
    print('PCA done.')
    x = TSNE(n_components=2, n_iter=5000).fit_transform(x)
elif tsne:
    x = TSNE(n_components=2, n_iter=5000).fit_transform(x)
elif pca:
    x = PCA(n_components=2).fit_transform(x)

print('TSNE done.')

colors = []
for i, r in enumerate(repr.values()):
    colors += [i for _ in range(r.shape[0])]

plt.scatter(*zip(*x), s=1, c=colors)
plt.show()

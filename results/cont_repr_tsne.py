import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

num_samples = 1000
tsne = True
pca = False
adv = False
file = '2018-11-20-105207' if adv else '2018-11-20-133729'

X = np.zeros((2*num_samples, 400))
i=0
with open(f'/home/fiit/{file}-pos-en') as f:
    for line in f:
        X[i] = np.array([float(n) for n in line.strip().split(', ')])
        i += 1
        if i == num_samples:
            break

with open(f'/home/fiit/{file}-pos-es') as f:
    for line in f:
        X[i] = np.array([float(n) for n in line.strip().split(', ')])
        i += 1
        if i == 2*num_samples:
            break

if tsne and pca:
    X = PCA(n_components=50).fit_transform(X)
    X = TSNE(n_components=2, n_iter=5000).fit_transform(X)
elif tsne:
    X = TSNE(n_components=2, n_iter=5000).fit_transform(X)
elif pca:
    X = PCA(n_components=2).fit_transform(X)

plt.scatter(*zip(*X), s=1, c=[0 for _ in range(num_samples)] + [1 for _ in range(num_samples)])
plt.show()

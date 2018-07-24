import os
import numpy as np

def dirs(path):
    return filter(lambda dir: os.path.isdir(path+'/'+dir), os.listdir(path))

def interweave(a, b):
    c = []
    for i in xrange(a.shape[0]):
        c.append(a[i])
        c.append(b[i])
    return np.array(c)

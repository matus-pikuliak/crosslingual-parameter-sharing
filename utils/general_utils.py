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

def mem_usage(msg=None):
    import os
    import psutil
    process = psutil.Process(os.getpid())
    if msg:
        print msg
    return process.memory_info().rss

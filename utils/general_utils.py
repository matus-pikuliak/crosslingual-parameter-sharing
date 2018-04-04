import os
import numpy as np

def dirs(path):
    return filter(lambda dir: os.path.isdir(path+'/'+dir), os.listdir(path))

def pad(sequence, length):
    return sequence.append(np.zeroes(length-len(sequence)))

import os


def directories(path):
    return filter(os.path.isdir, os.listdir(path))

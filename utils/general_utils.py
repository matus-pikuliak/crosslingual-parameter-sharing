import os


def dirs(path):
    return filter(os.path.isdir, os.listdir(path))

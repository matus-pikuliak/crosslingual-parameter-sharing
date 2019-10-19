import math
import os
import re
import subprocess
import sys
from collections import Iterable

import datetime


def dirs(path):
    return filter(lambda dir: os.path.isdir(path+'/'+dir), os.listdir(path))


def time_profile(msg=None):
    import os
    import psutil
    process = psutil.Process(os.getpid())
    if msg:
        print(msg)
    memory_now = process.memory_info().rss
    time_now = datetime.datetime.now()
    print(f"memory now: {memory_now:,}")
    print(f'time now: {time_now}')
    try:
        print(f"memory dif: {memory_now - time_profile.memory:,}")
    except:
        pass

    try:
        print(f'time dif: {time_now - time_profile.time}')
    except:
        pass

    time_profile.time = time_now
    time_profile.memory = memory_now
    print()


def add_hists(hists):
    final = dict(hists[0])
    for h in hists[1:]:
        for k, v in h.items():
            if k not in final:
                final[k] = 0
            final[k] += v
    return final


def f1(p, r):
    try:
        return 2*p*r/(p+r)
    except ZeroDivisionError:
        return 0


def git_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


def split_iter(string):
    return (x.group(0) for x in re.finditer(r"\S+", string))


def uneven_zip(*args):

    def try_next(ite):
        try:
            return next(ite)
        except StopIteration:
            return None

    args = [iter(arg) for arg in args]

    while True:
        output = [try_next(arg) for arg in args]
        output = list(filter(lambda x: x is not None, output))
        if output:
            yield output
        else:
            raise StopIteration


class RunningStd:
    """
    Welford's Online algorithm for calculating variance.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0
        self.m2 = 0

    def add(self, value):
        if isinstance(value, Iterable):
            for val in value:
                self.add(val)
        else:
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (value - self.mean)

    def std(self):
        return math.sqrt(self.m2 / self.n)

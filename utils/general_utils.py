import os
import numpy as np
import datetime

def dirs(path):
    return filter(lambda dir: os.path.isdir(path+'/'+dir), os.listdir(path))

def interweave(a, b):
    c = []
    for i in range(a.shape[0]):
        c.append(a[i])
        c.append(b[i])
    return np.array(c)

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
        print(f"memory dif: {memory_now - profile.memory:,}")
    except:
        pass

    try:
        print(f'time dif: {time_now - profile.time}')
    except:
        pass

    profile.time = time_now
    profile.memory = memory_now
    print()

def add_hists(hists):
    final = dict(hists[0])
    for h in hists[1:]:
        for k, v in h.items():
            if k not in final:
                final[k] = 0
            final[k] += v
    return final

# import itertools
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# data = [float(line) for line in open('oink').readlines()]
#
# plt.figure()
# intra = data[2::3] + data[1::3]
# inter = data[0::3]
# plt.plot([np.mean(intra[i::6]) for i in range(6)])
# plt.plot([np.mean(inter[i::6]) for i in range(6)])
# plt.plot(
#     np.array([np.mean(intra[i::6]) for i in range(6)])
#     - np.array([np.mean(inter[i::6]) for i in range(6)])
# )
# # plt.scatter([1, 2, 3, 4, 5, 6] * 20, )
# plt.show()
#
#
# exit()


import numpy as np

from get_representations import get_representations, get_desired, show_representations
from results.run_utils import init_runs, find_runs

log_path = '/home/fiit/logs/'
from results.run_db import db as run_db

runs = init_runs(log_path, run_db)
runs = find_runs(runs, name='zero-shot-two-by-two')

tasks = ['dep', 'lmo', 'ner', 'pos']
langs = ['cs', 'de', 'en', 'es']


def euclidean(a, b):
    count = 0.
    sum_ = 0.
    for ai in a:
        for bi in b:
            sum_ += np.sqrt(
                np.sum(
                    (ai - bi)**2
                ))
            count += 1
    return sum_ / count


for i, r in enumerate(runs[47:]):
    r.load()
    code = r.filename
    tl1, tl2 = r.config['tasks'][:2]
    task, lang = tl1
    _, epoch = r.metric_eval(task=task, language=lang)
    # print(epoch)
    # os.system(f'scp mpikuliak@147.175.145.128:/media/wd/mpikuliak/models/{code}-{epoch}*  /home/fiit/logs/models/')
    # des = get_desired(
    #     log_path+'deepnet2070/'+code,
    #     '-'.join(tl1),
    #     '-'.join(tl2))
    # print(des)
    #
    repr = show_representations(
        log_path+'deepnet2070/'+code,
        '/home/fiit/logs/models/'+code+'-'+str(epoch),
        tl1,
        tl2)
    # repr = list(repr.values())
    # with open('oink', 'a') as w:
    #     w.write(str(euclidean(repr[0], repr[1])) + '\n')
    #     w.write(str(euclidean(repr[0], repr[0])) + '\n')
    #     w.write(str(euclidean(repr[1], repr[1])) + '\n')

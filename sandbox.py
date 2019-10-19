import glob
import os

import numpy as np

from get_representations import show_representations
from results.run_utils import init_runs, find_runs

log_path = '/home/fiit/logs/'
model_path = '/home/fiit/logs/models/'
from results.run_db import db as run_db

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


runs = init_runs(log_path, run_db)
runs = find_runs(runs, name='zero-shot-adversarial-embs')

tasks = ['dep', 'lmo', 'ner', 'pos']
langs = ['cs', 'de', 'en', 'es']

for i, r in enumerate(runs):
    r.load()
    print(r.filename)
    tls = [('ner', 'cs'), ('ner', 'de'), ('ner', 'en'), ('ner', 'es')]
    task, lang = tls[0]
    try:
        _, epoch = r.metric_eval(task=task, language=lang)
    except:
        continue

    while True:
        if r.server == 'deepnet2070':
            os.system(f'scp mpikuliak@147.175.145.128:/media/wd/mpikuliak/models/{r.filename}-{epoch}*  {model_path}')
        if r.server == 'deepnet5':
            os.system(f'scp pikuliakm@147.175.145.178:/media/wd/pikuliakm/models/{r.filename}-{epoch}*  {model_path}')
        if glob.glob(f'/home/fiit/logs/models/{r.filename}-{epoch}*'):
            break
        epoch = int(epoch) // 5 * 5

    repr = show_representations(
        os.path.join(log_path, r.server, r.filename),
        os.path.join(model_path, r.filename + '-' + str(epoch)),
        *tls)
    break

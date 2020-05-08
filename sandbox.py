import ast
import glob
import os

import matplotlib.pyplot as plt
import tkinter

import numpy as np
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
plt_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

from get_representations import show_representations
from results.run_utils import init_runs, find_runs

log_path = '/home/mpikuliak/logs/'
model_path = '/home/mpikuliak/logs/models/'
from results.run_db import db as run_db

runs = init_runs(log_path, run_db)

tasks = ['dep', 'lmo', 'ner', 'pos']
langs = ['cs', 'de', 'en', 'es']


def get_run_data(run, lstm=True):
    run.load()
    print('Run file path', run.filename)
    tls = [('pos', l) for l in langs]
    task, lang = tls[0]
    _, epoch = run.metric_eval(task=task, language=lang)

    while True:
        if run.server == 'deepnet2070':
            os.system(f'scp mpikuliak@147.175.145.128:/media/wd/mpikuliak/models/{run.filename}-{epoch}*  {model_path}')
        if run.server == 'deepnet5':
            os.system(f'scp pikuliakm@147.175.145.178:/media/wd/pikuliakm/models/{run.filename}-{epoch}*  {model_path}')
        if glob.glob(f'/home/mpikuliak/logs/models/{run.filename}-{epoch}*'):
            break
        epoch = int(epoch) // 5 * 5 + 5

    return show_representations(
        os.path.join(log_path, run.server, run.filename),
        os.path.join(model_path, run.filename + '-' + str(epoch)),
        lstm=lstm,
        tls=tls)

runs = [
    find_runs(runs, name='zero-shot', type='ml-3', focus_on='pos-cs')[0],
    # find_runs(runs, name='zero-shot', type='all', focus_on='pos-cs')[0],
    find_runs(runs, name='zero-shot-task-lang', type='all', focus_on='pos-cs')[0],
    find_runs(runs, name='zero-shot-embs', type='all', focus_on='pos-cs')[0],
    find_runs(runs, name='zero-shot-adversarial', type='all', focus_on='pos-cs')[0],
]

fig, axes = plt.subplots(1, 4, squeeze=True)

for i, (c, run) in enumerate(zip('abcde', runs)):
    i += 3
    if i == 0:
        x, y, c1, c2 = get_run_data(run, lstm=False)
    else:
        x, y, c1, c2 = get_run_data(run)
    # res = [x, y, c1, c2]
    # with open(f'res_{i}', 'w') as f:
    #     f.write(str([list(r) for r in res]))
    # x, y, c1, c2 = ast.literal_eval(open(f'res_{i+1}').read())
    scatter = axes[i].scatter(x, y, c=c2, s=0.5)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].title.set_text(f'({c})')

    if i == 3:
        axes[i].legend(scatter.legend_elements()[0], langs, title="Languages", loc=1, bbox_to_anchor=(1.5, 0.5))

plt.show()

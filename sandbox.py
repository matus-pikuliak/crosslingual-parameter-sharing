import glob
import os

from get_representations import show_representations
from results.runs_db import db as runs_db
from results.run import Run
from get_representations import show_representations
from data.embedding import Embeddings


task_metr = {
    'dep': 'las',
    'lmo': 'perplexity',
    'ner': 'chunk_f1',
    'pos': 'acc'
}

task_max = {
    'dep': True,
    'lmo': False,
    'ner': True,
    'pos': True
}

log_path = '/home/mpikuliak/logs'
runs = []

for server in runs_db:
    paths = glob.glob(os.path.join(log_path, server, '*'))
    paths = iter(sorted(paths))

    try:
        for (number, type_, code) in runs_db[server]:
            for _ in range(number):
                try:
                    path = next(paths)
                    runs.append(Run(path, type_, code))
                except KeyError:
                    print(path)
    except StopIteration:
        pass


def find_runs(run_code=None, run_type=None, contains=None, **config):
    if contains is None:
        contains = []

    return (run
            for run
            in runs
            if (run_code is None or run_code == run.code) and
            (run_type is None or run_type == run.type) and
            all(run.contains(*task_lang) for task_lang in contains) and
            all(run.config[key] == value for key, value in config.items()))


rs = find_runs(run_code='zero-shot-two-by-two')

for i, r in enumerate(rs):
    if i >= 51:
        Embeddings.cache = {}
        path = '/home/mpikuliak/logs/deepnet2070/'
        code = r.name
        r = Run(path+code, None, None)
        tl1, tl2 = r.config['tasks'][:2]
        task, lang = tl1
        _, epoch = r.metric_eval(task_metr[task], max_=task_max[task], task=task, language=lang)
        os.system(f'scp mpikuliak@147.175.145.128:/media/wd/mpikuliak/models/{code}-{epoch}*  /home/mpikuliak/logs/models/')
        print(r.name)
        show_representations(
            path+code,
            '/home/mpikuliak/logs/models/'+code+'-'+str(epoch),
            '-'.join(tl1),
            '-'.join(tl2))
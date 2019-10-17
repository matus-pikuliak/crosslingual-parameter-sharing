import glob
import itertools
import os

from results.run import Run


def init_runs(log_path, run_db):
    runs = []

    for server, records in run_db.items():
        paths = glob.glob(os.path.join(log_path, server, '*'))
        paths = sorted(paths)

        records = [[(type, name)] * number for (number, type, name) in records]
        records = list(itertools.chain.from_iterable(records))
        records = iter(records)

        for path in paths:
            record = next(records)
            run = Run(path, *record, autoload=False)
            runs.append(run)

    return runs


def find_runs(runs, type=None, name=None, contains=(), **config):
    return [run for run in runs if run.match(type, name, contains, **config)]

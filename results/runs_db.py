db = {
    'deepnet5': [
        (8, 'var', 'early-5'),
        (8, 'var', 'early-10'),
        (8, 'var', 'early-no'),
        (36, 'ml-1', 'zero-shot'),
        (12, 'rel', 'zero-shot'),
        (12, 'rel', 'zero-shot-task-lang'),
        (12, 'rel', 'zero-shot-task-lang-no-global'),
        (12, 'all', 'zero-shot-task-lang-ortho-50'),
        (12, 'all', 'zero-shot-task-lang-no-global-ortho-50'),
        (12, 'all', 'zero-shot-task-lang-ortho-100'),
        (12, 'all', 'zero-shot-task-lang-no-global-ortho-100'),
        (12, 'all', 'zero-shot-task-lang-ortho-200'),
        (12, 'all', 'zero-shot-task-lang-no-global-ortho-200'),
        
    ],
    'deepnet2070': [
        (12, 'ml-3', 'zero-shot'),
        (12, 'all', 'zero-shot'),
        (12, 'all', 'zero-shot-task-lang'),
        (12, 'all', 'zero-shot-task-lang-no-global'),
    ],
}

print([(server, sum(tpl[0] for tpl in tuples)) for server, tuples in db.items()])

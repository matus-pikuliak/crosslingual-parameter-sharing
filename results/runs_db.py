db = {
    'deepnet5': [
        (8, 'var', 'early-5'),
        (8, 'var', 'early-10'),
        (8, 'var', 'early-no'),
        (36, 'ml', 'zero-shot'),
    ]
}

print([(server, sum(tpl[0] for tpl in tuples)) for server, tuples in db.items()])

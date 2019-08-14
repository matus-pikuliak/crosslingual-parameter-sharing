db = {
    'deepnet5': [
        (8, 'var', 'early-5'),
        (8, 'var', 'early-10'),
        (8, 'var', 'early-no'),
    ]
}

print([(server, sum(tpl[0] for tpl in tuples)) for server, tuples in db.items()])

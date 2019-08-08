db = {
    'deepnet5': [
        (0, 'top3', 'private-focused'),
    ]
}

print([(server, sum(tpl[0] for tpl in tuples)) for server, tuples in db.items()])

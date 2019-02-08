db = {
    'martak': [
        (16, 'stsl', 'vanilla'),
        (4, 'ml', 'vanilla'),
        (4, 'mt', 'vanilla'),
        (1, 'mtml', 'vanilla'),
        (1, 'mtml', 'no-adv'),
        (1, 'mtml', 'no-task-sharing'),
        (4, 'ml', 'no-mwe'),
        (1, 'mtml', 'no-mwe'),
        (16, 'stsl', 'vanilla400'),
        (4, 'ml', 'vanilla400'),
        (4, 'mt', 'vanilla400'),
        (1, 'mtml', 'vanilla400'),
        (4, 'ml', 'no-adv-task-sharing'),
        (1, 'mtml', 'no-adv-task-sharing'),
        (1, 'ml', 'private-old'),
        (4, 'ml', 'private'),
        #(1, 'mtml', 'private'),
        (1, 'ml', 'dep-adv-lambda-0.01'),
        (1, 'ml', 'dep-private-adv-lambda-0.125'),
    ],
    'gcp': [
        (4, 'ml', 'no-adv'),
        (4, 'ml', 'no-task-sharing'),
        (4, 'ml', 'no-adv-mwe'),
        (1, 'mtml', 'no-adv-mwe'),
        (16, 'stsl', 'vanilla300'),
        (4, 'ml', 'vanilla300'),
        (4, 'mt', 'vanilla300'),
        (1, 'mtml', 'vanilla300'),
        (1, 'ml', 'dep-adv-lambda-0.25'),
        (1, 'ml', 'dep-adv-lambda-0.125'),
        (1, 'ml', 'dep-adv-freq-2'),
        (4, 'mt', 'private'),
    ],
    'deepnet2': [
        (4, 'ml', 'private-with-adv'),
        (4, 'mt', 'private-with-adv'),
        # (1, 'mtml', 'private-with-adv'),
    ]
}

print([(server, sum(tpl[0] for tpl in tuples)) for server, tuples in db.items()])

# vanilla focused
# private mtml (without irrelevant) focused
# 200 2000 ml mt vanilla?
# pick best and do fine tuning
# zero shot for new languages - slovak, finnish, portugese, italian?

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
        (1, 'ml', 'dep-adv-lambda-0.01'),
        (1, 'ml', 'dep-private-adv-lambda-0.25'),
        (4, 'mt', 'private-focused-dep-ner-pos-cs-es'),
        (4, 'mt', 'private-focused-0.75-dep-ner-pos-cs-es'),
        (6, 'stsl', 'vanilla-2000'),
        (6, 'stsl', 'vanilla-200'),
        (3, 'mtml', 'private-focused-0.75-dep-ner-pos-es'),  # without cs
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
        (6, 'mt', 'private-focused-dep-ner-pos-cs-es-2000'),
        (6, 'mt', 'private-focused-dep-ner-pos-cs-es-200'),
    ],
    'deepnet2': [
        (4, 'ml', 'private-with-adv'),
        (4, 'mt', 'private-with-adv'),
        (6, 'ml', 'private-focused-dep-ner-pos-cs-es'),
        (6, 'ml', 'private-focused-0.75-dep-ner-pos-cs-es'),
        (1, 'mtml', 'private'),
        (1, 'mtml', 'private-with-adv'),
        (6, 'ml', 'private-focused-dep-ner-pos-cs-es-2000'),
        (6, 'ml', 'private-focused-dep-ner-pos-cs-es-200'),
        (2, 'mt', 'private-focused-dep-ner-pos-cs-es'),  # martak bug
        (2, 'mt', 'private-focused-0.75-dep-ner-pos-cs-es'),
    ],
    'acer': [
        (3, 'ml', 'dep-cs-one-aux'),
        (3, 'mt', 'dep-cs-one-aux')
    ]
}

print([(server, sum(tpl[0] for tpl in tuples)) for server, tuples in db.items()])

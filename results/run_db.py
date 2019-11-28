
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
        (12, 'all', 'zero-shot-task'),
        (12, 'all', 'zero-shot-lang'),
        (12, 'all', 'zero-shot-task-lang-ortho-25'),
        (12, 'all', 'zero-shot-task-lang-ortho-10'),
        (12, 'all', 'zero-shot-embs-rotated'),
        (12, 'ml-3', 'zero-shot-rotated'),
        (12, 'all', 'zero-shot-char-level'),
        (12, 'all', 'zero-shot-embs-char-level'),
        (12, 'all', 'zero-shot-embs-400-char-level'),
        (12, 'all', 'zero-shot-embs-400-rotated'),
        (12, 'all', 'zero-shot-400'),
        (12, 'all', 'zero-shot-adversarial-task-lang'),
        (12, 'all', 'zero-shot-embs-limited-task-200'),
        (12, 'all', 'zero-shot-embs-limited-lang-200'),
        (12, 'all', 'zero-shot-task-lang-limited-task-200'),
        (12, 'all', 'zero-shot-task-lang-limited-lang-200'),
        (12, 'all', 'zero-shot-char-level-adversarial'),
        (50, 'var', 'low-resource-2'),  # 10 target-source parov, pre kazdy Single, ML, MT, ML+MT, ALL
        (7, 'var', 'no-dropout'),  # bug dropout, novy dropout
        (12, 'all', 'zero-shot-task-lang-ortho-50-again'),
        (12, 'rel', 'zero-shot-rel-again'),
        (12, 'rel', 'zero-shot-adv'),
        (12, 'all', 'zero-shot-adversarial-task-emb'),
        (12, 'rel', 'low-resource'),
    ],
    'deepnet2070': [
        (12, 'ml-3', 'zero-shot'),
        (12, 'all', 'zero-shot'),
        (12, 'all', 'zero-shot-task-lang'),
        (12, 'all', 'zero-shot-task-lang-no-global'),
        (12, 'all', 'zero-shot-embs'),
        (12, 'rel', 'zero-shot-embs'),
        (12, 'all', 'zero-shot-embs-400'),
        (12, 'rel', 'zero-shot-embs-400'),
        (12, 'stsl', 'normal-training'),
        (12, 'all', 'zero-shot-task-emb'),
        (12, 'all', 'zero-shot-lang-emb'),
        (12, 'all', 'zero-shot-task-lang-both-embs'),
        (12, 'all', 'zero-shot-rotated'),
        (12, 'all', 'zero-shot-task-lang-rotated'),
        (12, 'ml-3', 'zero-shot-char-level'),
        (12, 'all', 'zero-shot-task-lang-char-level'),
        (120, 'var', 'zero-shot-two-by-two'),  # 20 target-source parov, pre kazdy ML, REL, ML+UNREL, ALL, ALL+emb, ALL+params
        (12, 'all', 'zero-shot-adversarial'),
        (12, 'all', 'zero-shot-adversarial-embs'),
        (12, 'all', 'zero-shot-limited-task-200'),
        (12, 'all', 'zero-shot-limited-lang-200'),
        (12, 'all', 'zero-shot-adversarial-limited-task-200'),
        (12, 'all', 'zero-shot-adversarial-limited-lang-200'),
        (20, 'var', 'zero-shot-two-by-two-adversarial'),  # adversarial doplnok ku 120 runom vyssie
        (6, 'frob', 'zero-shot-dep-cs-frobenius'),  # hparam tuning
        (12, 'all', 'zero-shot-rotated-adversarial'),
        (50, 'var', 'low-resource'),  # 10 target-source parov, pre kazdy Single, ML, MT, ML+MT, ALL
        (4, 'var', 'no-dropout'),  # iny dropout zle, povodny dropout
        (60, 'var', 'low-resource-advanced'),  # 20 target-source parov, adversarial, embs, parameter
        (12, 'all', 'zero-shot-adv-again'),  # 20 target-source parov, adversarial, embs, parameter
        (12, 'ml-unrel-12', 'zero-shot'),
        (12, 'rel', 'zero-shot-embs'),
        (12, 'all', 'zero-shot-adversarial-task-para'),
    ],
    'deepnet6-1': [
        (12, 'ml-3', 'low-resource'),
        (12, 'mt-3', 'low-resource'),
    ],
    'deepnet6-2': [
        (12, 'all', 'low-resource'),
    ],
}
print([(server, sum(tpl[0] for tpl in tuples)) for server, tuples in db.items()])

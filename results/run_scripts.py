import itertools

import numpy as np

ls = ['cs', 'de', 'en', 'es']
ts = ['dep', 'lmo', 'ner', 'pos']

'''
27.10.2019 deepnet2070 (first 10) / deepnet 5 (second 10)
low-resource 200
'''

fours = [
    ['pos-es', 'pos-cs', 'dep-es', 'dep-cs'],
    ['pos-de', 'pos-es', 'ner-de', 'ner-es'],
    ['pos-es', 'pos-cs', 'lmo-es', 'lmo-cs'],
    ['dep-cs', 'dep-en', 'ner-cs', 'ner-en'],
    ['ner-en', 'ner-de', 'dep-en', 'dep-de'],
    ['dep-es', 'dep-de', 'ner-es', 'ner-de'],
    ['dep-cs', 'dep-es', 'lmo-cs', 'lmo-es'],
    ['ner-en', 'ner-cs', 'dep-en', 'dep-cs'],
    ['ner-es', 'ner-cs', 'lmo-es', 'lmo-cs'],
    ['pos-cs', 'pos-en', 'ner-cs', 'ner-en'],
    ['ner-es', 'ner-en', 'dep-es', 'dep-en'],
    ['pos-cs', 'pos-de', 'ner-cs', 'ner-de'],
    ['pos-cs', 'pos-es', 'dep-cs', 'dep-es'],
    ['dep-es', 'dep-de', 'lmo-es', 'lmo-de'],
    ['dep-en', 'dep-es', 'pos-en', 'pos-es'],
    ['dep-en', 'dep-es', 'ner-en', 'ner-es'],
    ['ner-en', 'ner-es', 'dep-en', 'dep-es'],
    ['pos-en', 'pos-es', 'lmo-en', 'lmo-es'],
    ['pos-de', 'pos-en', 'dep-de', 'dep-en'],
    ['dep-cs', 'dep-es', 'pos-cs', 'pos-es'],
]

for four in fours[10:]:
    print(
        f'bash train.sh focus_on {four[0]} limited_task_language {four[0]} limited_data_size 200 task_layer_private false epochs 100 early_stopping 10 tasks {four[0]}',
        end=' && ')
    print(
        f'bash train.sh focus_on {four[0]} limited_task_language {four[0]} limited_data_size 200 task_layer_private false epochs 100 early_stopping 10 tasks {four[0]} {four[1]}',
        end=' && ')
    print(
        f'bash train.sh focus_on {four[0]} limited_task_language {four[0]} limited_data_size 200 task_layer_private false epochs 100 early_stopping 10 tasks {four[0]} {four[2]}',
        end=' && ')
    print(
        f'bash train.sh focus_on {four[0]} limited_task_language {four[0]} limited_data_size 200 task_layer_private false epochs 100 early_stopping 10 tasks {four[0]} {four[1]} {four[2]}',
        end=' && ')
    print(
        f'bash train.sh focus_on {four[0]} limited_task_language {four[0]} limited_data_size 200 task_layer_private false epochs 100 early_stopping 10 tasks {" ".join(four)}',
        end=' && ')

exit()

'''
27.10.2019 deepnet5
zero-shot adversarial char-level
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} adversarial_training True char_level True word_level False focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()

'''
27.10.2019 deepnet2070
zero-shot adversarial mwe_rotated
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} adversarial_training True word_emb_type mwe_rotated focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()


'''
25.10.2019 deepnet2070
zero-shot task-lang with frobenius
'''
for frob in [0.01, 0.1, 1.0, 0.001, 0.0001, 0.0]:
        print(
            f'bash train.sh focus_on dep-cs frobenius {frob} word_lstm_lang true word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
            end=' && ')
exit()

'''
24.10.2019 deepnet2070
zero-shot 2x2 with adversarial
'''
fours = [
    ['pos-es', 'pos-cs', 'dep-es', 'dep-cs'],
    ['pos-de', 'pos-es', 'ner-de', 'ner-es'],
    ['pos-es', 'pos-cs', 'lmo-es', 'lmo-cs'],
    ['dep-cs', 'dep-en', 'ner-cs', 'ner-en'],
    ['ner-en', 'ner-de', 'dep-en', 'dep-de'],
    ['dep-es', 'dep-de', 'ner-es', 'ner-de'],
    ['dep-cs', 'dep-es', 'lmo-cs', 'lmo-es'],
    ['ner-en', 'ner-cs', 'dep-en', 'dep-cs'],
    ['ner-es', 'ner-cs', 'lmo-es', 'lmo-cs'],
    ['pos-cs', 'pos-en', 'ner-cs', 'ner-en'],
    ['ner-es', 'ner-en', 'dep-es', 'dep-en'],
    ['pos-cs', 'pos-de', 'ner-cs', 'ner-de'],
    ['pos-cs', 'pos-es', 'dep-cs', 'dep-es'],
    ['dep-es', 'dep-de', 'lmo-es', 'lmo-de'],
    ['dep-en', 'dep-es', 'pos-en', 'pos-es'],
    ['dep-en', 'dep-es', 'ner-en', 'ner-es'],
    ['ner-en', 'ner-es', 'dep-en', 'dep-es'],
    ['pos-en', 'pos-es', 'lmo-en', 'lmo-es'],
    ['pos-de', 'pos-en', 'dep-de', 'dep-en'],
    ['dep-cs', 'dep-es', 'pos-cs', 'pos-es'],
]

for f in fours:
    print(
        f'bash train.sh focus_on {f[0]} adversarial_training True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {" ".join(f)}',
        end=' && ')

exit()

'''
21.10.2019 deepnet5
zero-shot all-to-one limited task lang 200 embs
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_task True word_lstm_lang True limited_task {t} limited_data_size 200 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_task True word_lstm_lang True limited_language {l} limited_data_size 200 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()

'''
21.10.2019 deepnet5
zero-shot all-to-one limited task lang 200 embs
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} emb_task True emb_lang True limited_task {t} limited_data_size 200 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} emb_task True emb_lang True limited_language {l} limited_data_size 200 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()

'''
21.10.2019 deepnet2070
zero-shot all-to-one limited task lang 200 adversarial
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} adversarial_training True limited_task {t} limited_data_size 200 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} adversarial_training True limited_language {l} limited_data_size 200 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()


'''
19.10.2019 deepnet2070
zero-shot all-to-one limited task lang 200
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} limited_task {t} limited_data_size 200 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} limited_language {l} limited_data_size 200 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()

'''
17.10.2019 deepnet5
adversarial + task/lang
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} adversarial_training True word_lstm_task True word_lstm_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()

'''
17.10.2019 deepnet2070
adversarial + embs
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} adversarial_training True emb_task True emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()
'''
11.10.2019 deepnet2070
adversarial
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} adversarial_training True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

print()

'''
11.10.2019 deepnet5
lstm 400
'''
for t in ts:
    print(t)
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_size 400 focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
exit()

pairs = set()
while len(pairs) < 20:
    tgt_task = np.random.choice(list(ts - {'lmo'}))
    tgt_lang = np.random.choice(list(ls))
    src_task = np.random.choice(list(ts - {tgt_task}))
    src_lang = np.random.choice(list(ls - {tgt_lang}))
    tpl = (tgt_task, tgt_lang, src_task, src_lang)
    if tpl not in pairs:
        pairs.add(tpl)
    else:
        continue

    print(
        f'bash train.sh focus_on {tgt_task}-{tgt_lang} focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tgt_task}-{tgt_lang} {tgt_task}-{src_lang}',
        end=' && ')
    print(
        f'bash train.sh focus_on {tgt_task}-{tgt_lang} focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tgt_task}-{tgt_lang} {tgt_task}-{src_lang} {src_task}-{tgt_lang}',
        end=' && ')
    print(
        f'bash train.sh focus_on {tgt_task}-{tgt_lang} focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tgt_task}-{tgt_lang} {tgt_task}-{src_lang} {src_task}-{src_lang}',
        end=' && ')
    print(
        f'bash train.sh focus_on {tgt_task}-{tgt_lang} focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tgt_task}-{tgt_lang} {tgt_task}-{src_lang} {src_task}-{tgt_lang} {src_task}-{src_lang}',
        end=' && ')
    print(
        f'bash train.sh focus_on {tgt_task}-{tgt_lang} emb_task True emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tgt_task}-{tgt_lang} {tgt_task}-{src_lang} {src_task}-{tgt_lang} {src_task}-{src_lang}',
        end=' && ')
    print(
        f'bash train.sh focus_on {tgt_task}-{tgt_lang} word_lstm_lang True word_lstm_task True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tgt_task}-{tgt_lang} {tgt_task}-{src_lang} {src_task}-{tgt_lang} {src_task}-{src_lang}',
        end=' && ')

exit()
'''
24.09.2019 deepnet5 run
zero shot random MWE, char level both with 400 embs
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_size 400 word_emb_type mwe_rotated emb_task True emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_size 400 word_level false char_level true emb_lang true emb_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
exit()

'''
24.09.2019 deepnet5 run
zero shot char level all&& embs
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_level false char_level true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_level false char_level true emb_lang true emb_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
exit()
'''
24.09.2019 deepnet2070 run
zero shot char level ml-3 && task-lang all
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            tasks = ' '.join([f'{t}-{l2}' for l2 in ls])
            print(
                f'bash train.sh focus_on {t}-{l} word_level false char_level true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tasks}',
                end=' && ')

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_level false char_level true word_lstm_lang true word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
exit()

'''
21.09.2019 deepnet5 run
zero shot random MWE
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            tasks = ' '.join([f'{t}-{l2}' for l2 in ls])
            print(
                f'bash train.sh focus_on {t}-{l} word_emb_type mwe_rotated focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tasks}',
                end=' && ')
exit()

'''
20.09.2019 deepnet2070 run
zero shot random MWE
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_emb_type mwe_rotated focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_emb_type mwe_rotated word_lstm_lang true word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_emb_type mwe_rotated emb_task True emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()
'''
04.09.2019 deepnet5 run
zero shot task or lang sharing
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_lang true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} ortho 25 word_lstm_lang true word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} ortho 10 word_lstm_lang true word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
exit()
'''
04.09.2019 deepnet2070 run
zero shot task or lang embeddings
'''

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} emb_task True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} emb_task true emb_lang true word_lstm_lang true word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')

exit()

'''
02.09.2019 deepnet2070 run
normal baselines
'''

for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh epochs 100 early_stopping 10 tasks {t}-{l}',
                end=' && ')
exit()


'''
24.08.2019 deepnet2070 run
zero-shot all rel transfer with task and lang embeddings
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} emb_task True emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            tasks = ' '.join([f'{t2}-{l2}' for t2 in ts for l2 in ls if t == t2 or l == l2])
            print(
                f'bash train.sh focus_on {t}-{l} emb_task True emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tasks}',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_size 400 emb_task True emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            tasks = ' '.join([f'{t2}-{l2}' for t2 in ts for l2 in ls if t == t2 or l == l2])
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_size 400 emb_task True emb_lang True focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tasks}',
                end=' && ')
exit()


'''
15.08.2019 deepnet2070 run
zero-shot all transfer many-to-one
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_lang true word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_lang true word_lstm_task true  word_lstm_global false focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks all',
                end=' && ')
exit()

'''
15.08.2019 deepnet5 run
zero-shot related transfer many-to-one
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            tasks = ' '.join([f'{t2}-{l2}' for t2 in ts for l2 in ls if t == t2 or l == l2])
            print(
                f'bash train.sh focus_on {t}-{l} focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tasks}',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            tasks = ' '.join([f'{t2}-{l2}' for t2 in ts for l2 in ls if t == t2 or l == l2])
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_lang true word_lstm_task true focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tasks}',
                end=' && ')
for t in ts:
    for l in ls:
        if t != 'lmo':
            tasks = ' '.join([f'{t2}-{l2}' for t2 in ts for l2 in ls if t == t2 or l == l2])
            print(
                f'bash train.sh focus_on {t}-{l} word_lstm_lang true word_lstm_task true  word_lstm_global false focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tasks}',
                end=' && ')
exit()


'''
15.08.2019 deepnet2070 run
zero-shot ML transfer many-to-one
'''
for t in ts:
    for l in ls:
        if t != 'lmo':
            tasks = ' '.join([f'{t}-{l2}' for l2 in ls])
            print(
                f'bash train.sh focus_on {t}-{l} focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {tasks}',
                end=' && ')
exit()

'''
14.08.2019 deepnet5 run
zero-shot ML transfer one-to-one
'''
for t in ts:
    for l in ls:
        for l2 in ls:
            if l2 != l and t != 'lmo':
                print(
                    f'bash train.sh focus_on {t}-{l} focus_rate 0 task_layer_private false epochs 100 early_stopping 10 tasks {t}-{l} {t}-{l2}',
                    end=' && ')
exit()






# TOP1
# TOP3 pre pos-de     'pos-de': ['ner-de', 'pos-en', 'lmo-de'],
'''
20.05.2019 deepnet5 run
'''
tops = {
    'ner-en': 'ner-cs',
    'ner-es': 'lmo-es',
    'pos-cs': 'dep-cs',
    'pos-de': 'ner-de',
    'pos-en': 'lmo-en',
    'pos-es': 'pos-en',
    'dep-cs': 'dep-es',
    'dep-de': 'dep-cs',
    'dep-en': 'dep-es',
    'dep-es': 'dep-en',
    'ner-cs': 'dep-cs',
    'ner-de': 'lmo-de',
}

for k, v in tops.items():
    print(
        f'bash train.sh task_layer_sharing false adversarial_training false private_params true focus_rate 0.5 focus_on {k} tasks {k} {v}',
        end=' && ')
exit()

'''
20.05.2019 gcp run
'''
tops = {
    'dep-cs': 'dep-es',
    'dep-de': 'dep-cs',
    'dep-en': 'dep-es',
    'dep-es': 'dep-en',
    'ner-cs': 'dep-cs',
    'ner-de': 'lmo-de',
}

for k, v in tops.items():
    print(
        f'bash train.sh task_layer_sharing false adversarial_training false private_params true focus_rate 0.5 focus_on {k} tasks {k} {v}',
        end=' && ')
exit()


'''
20.05.2019 deepnet2070 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['de', 'en']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true focus_rate 0.75 focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            for lt in ls:
                if lt == l or tt == t:
                    print(f'{tt}-{lt} ', end='')
        print('&& ', end='')
exit()


'''
19.05.2019 deepnet5 run
'''
ll = ['de', 'en']
tt = ['dep', 'ner', 'pos']
setup = {f'{t}-{l}': [f'{t2}-{l2}' for t2, l2 in itertools.product(ts, ls) if t2 == t] for t, l in itertools.product(tt, ll)}

for k, v in setup.items():
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_rate 0.5 focus_on {k} tasks {" ".join(v)}', end=' ')
        print('&&', end=' ')

exit()


'''
18.05.2019 deepnet2070 run
'''
ll = ['de', 'en']
tt = ['dep', 'ner', 'pos']
setup = {f'{t}-{l}': [f'{t2}-{l2}' for t2, l2 in itertools.product(ts, ls) if l2 == l] for t, l in itertools.product(tt, ll)}

for k, v in setup.items():
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_rate 0.5 focus_on {k} tasks {" ".join(v)}', end=' ')
        print('&&', end=' ')

exit()

'''
17.05.2019 deepnet5 run
'''
setup = {
    'ner-cs': ['dep-cs', 'ner-en', 'pos-cs'],
    'ner-es': ['lmo-es', 'ner-de', 'ner-cs'],
    'dep-de': ['dep-cs', 'dep-es', 'dep-en'],
    'dep-en': ['dep-es', 'dep-cs', 'lmo-en'],
    'ner-de': ['lmo-de', 'ner-cs', 'ner-en'],
    'ner-en': ['ner-cs', 'ner-es', 'pos-en'],
    'pos-de': ['ner-de', 'pos-de', 'lmo-de'],
    'pos-en': ['lmo-en', 'ner-en', 'dep-en']
}

for k, v in setup.items():
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_rate 0.5 focus_on {k} tasks {" ".join([k] + v)}', end=' ')
        print('&&', end=' ')

exit()


'''
16.05.2019 deepnet2070 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false focus_rate 0.866 focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            for lt in ls:
                if lt == l or tt == t:
                    print(f'{tt}-{lt} ', end='')
        print('&& ', end='')
exit()


'''
28.02.2019 deepnet2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs']:
        print(f'bash train.sh private_params true  task_layer_sharing false adversarial_training false limited_data_size 200 limited_task_language {t}-{l} tasks ', end='')
        for lt in ls:
            print(f'{t}-{lt} ', end='')
        print('&& ', end='')
exit()

'''
28.02.2019 martak run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false limited_data_size 200 limited_task_language {t}-{l} tasks ', end='')
        for lt in ls:
            print(f'{t}-{lt} ', end='')
        print('&& ', end='')
exit()

'''
28.02.2019 deepnet2 run
'''

for t in ['dep', 'ner', 'pos']:
    for l in ['es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false focus_on {t}-{l} focus_rate 0.75 tasks ', end='')
        for lt in ls:
            print(f'{t}-{lt} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
26.02.2019 fiit-gcp-2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false focus_on {t}-{l} focus_rate 0.75 tasks ', end='')
        for lt in ls:
            print(f'{t}-{lt} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
26.02.2019 deepnet2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false focus_on {t}-{l} focus_rate 0.75 tasks ', end='')
        for tt in ts:
            print(f'{tt}-{l} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()


'''
25.02.2019 deepnet2 run
'''
int_task = ('ner', 'en')
for (t, l) in [('ner', 'de'), ('lmo', 'en'), ('pos', 'en')]:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ner-en {t}-{l}', end=' && ')
int_task = ('ner', 'de')
for (t, l) in [('lmo', 'de'), ('pos', 'de')]:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ner-de {t}-{l}', end=' && ')
int_task = ('pos', 'en')
for (t, l) in [('pos', 'de'), ('lmo', 'en')]:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks pos-en {t}-{l}', end=' && ')
int_task = ('pos', 'de')
for (t, l) in [('lmo', 'de')]:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks pos-de {t}-{l}', end=' && ')
exit()

'''
25.02.2019 fiit-gcp-3 run
'''
int_task = ('dep', 'en')
for (t, l) in [('dep', 'de'), ('lmo', 'en'), ('ner', 'en'), ('pos', 'en')]:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks dep-en {t}-{l}', end=' && ')
int_task = ('dep', 'de')
for (t, l) in [('lmo', 'de'), ('ner', 'de'), ('pos', 'de')]:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks dep-de {t}-{l}', end=' && ')
print('; sudo poweroff', end='')
exit()

'''
25.02.2019 fiit-gcp-1 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            for lt in ls:
                if (tt == t) or (lt ==l):
                    print(f'{tt}-{lt} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
24.02.2019 gcp run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            for lt in ls:
                if (tt == t) or (lt ==l):
                    print(f'{tt}-{lt} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
23.02.2019 martak run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false limited_data_size 200 limited_task_language {t}-{l} tasks ', end='')
        for tt in ts:
            for lt in ls:
                if (tt == t) or (lt ==l):
                    print(f'{tt}-{lt} ', end='')
        print('&& ', end='')
exit()


'''
23.02.2019 deepnet2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false limited_data_size 2000 limited_task_language {t}-{l} tasks ', end='')
        for tt in ts:
            for lt in ls:
                if (tt == t) or (lt ==l):
                    print(f'{tt}-{lt} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
22.02.2019 martak run
'''
for l in ls:
    print(f'bash train.sh task_layer_sharing false adversarial_training false tasks ', end='')
    for t in ['dep', 'ner', 'pos']:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
exit()


'''
22.02.2019 deepnet2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false limited_data_size 200 limited_task_language {t}-{l} tasks ', end='')
        for tt in ts:
            print(f'{tt}-{l} ', end='')
        print('&& ', end='')
exit()

'''
21.02.2019 fiit-gcp-2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false limited_data_size 2000 limited_task_language {t}-{l} tasks ', end='')
        for tt in ts:
            print(f'{tt}-{l} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()



'''
20.02.2019 fiit-gcp-3 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false limited_data_size 200 limited_task_language {t}-{l} tasks ', end='')
        for lt in ls:
            print(f'{t}-{lt} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()


'''
20.02.2019 fiit-gcp-1 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false limited_data_size 2000 limited_task_language {t}-{l} tasks ', end='')
        for lt in ls:
            print(f'{t}-{lt} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
20.02.2019 deepnet2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            print(f'{tt}-{l} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()


'''
20.02.2019 gcp run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh task_layer_sharing false adversarial_training false focus_on {t}-{l} tasks ', end='')
        for l in ls:
            print(f'{t}-{l} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()


'''
19.02.2019 fiit-gcp-1,2,3 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_rate 0.866 focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            for lt in ls:
                if lt == l or tt == t:
                    print(f'{tt}-{lt} ', end='')
        print('; sudo shutdown')
exit()


'''
19.02.2019 martak run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_rate 0.866 focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            for lt in ls:
                if lt == l or tt == t:
                    print(f'{tt}-{lt} ', end='')
        print('&& ', end='')
exit()

'''
18.02.2019 deepnet run
dep-es/ner-es oprava z martak 11.02
'''
for t in ['dep', 'ner']:
    for l in ['es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            print(f'{tt}-{l} ', end='')
        print('&& ', end='')
for t in ['dep', 'ner']:
    for l in ['es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} focus_rate 0.75 tasks ', end='')
        for tt in ts:
            print(f'{tt}-{l} ', end='')
        print('&& ', end='')
exit()


'''
18.02.2019 martak run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh limited_data_size 2000 limited_task_language {t}-{l} tasks {t}-{l}', end=' && ')
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh limited_data_size 200 limited_task_language {t}-{l} tasks {t}-{l}', end=' && ')
exit()

'''
17.02.2019 acer run
'''
int_task = ('dep', 'es')
for (t, l) in itertools.product(ts, ls):
    if (t == int_task[0]) ^ (l == int_task[1]):
        print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks dep-es {t}-{l}', end=' && ')
int_task = ('ner', 'cs')
for (t, l) in itertools.product(['lmo', 'ner', 'pos'], ls):
    if (t == int_task[0]) ^ (l == int_task[1]):
        print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ner-cs {t}-{l}', end=' && ')
int_task = ('ner', 'es')
for (t, l) in itertools.product(['lmo', 'ner', 'pos'], ls):
    if (t == int_task[0]) ^ (l == int_task[1]):
        print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ner-es {t}-{l}', end=' && ')
int_task = ('pos', 'cs')
for (t, l) in itertools.product(['lmo', 'pos'], ls):
    if (t == int_task[0]) ^ (l == int_task[1]):
        print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks pos-cs {t}-{l}', end=' && ')
int_task = ('pos', 'es')
for (t, l) in itertools.product(['lmo', 'pos'], ls):
    if (t == int_task[0]) ^ (l == int_task[1]):
        print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks pos-es {t}-{l}', end=' && ')
exit()

'''
14.02.2019 deepnet2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} limited_data_size 2000 limited_task_language {t}-{l} tasks ', end='')
        for l in ls:
            print(f'{t}-{l} ', end='')
        print('&& ', end='')
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} limited_data_size 200 limited_task_language {t}-{l} tasks ', end='')
        for l in ls:
            print(f'{t}-{l} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()


'''
14.02.2019 acer run
'''
int_task = ('dep', 'cs')
for (t, l) in itertools.product(ts, ls):
    if (t == int_task[0]) ^ (l == int_task[1]):
        print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks dep-cs {t}-{l}', end=' &&')
exit()


'''
12.02.2019 deepnet run
'''
print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
print(f'&& ', end='')
print(f'bash train.sh task_layer_sharing false private_params true tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
exit()


'''
11.02.2019 martak run
dep-es/ner-es kvoli bugu opat spustene na deepnete 18.02
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} tasks ', end='')
        for tt in ts:
            print(f'{tt}-{l} ', end='')
        print('&& ', end='')
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} focus_rate 0.75 tasks ', end='')
        for tt in ts:
            print(f'{tt}-{l} ', end='')
        print('&& ', end='')
exit()

'''
11.02.2019 gcp run
'''

for l in ls:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ', end='')
    for t in ts:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} limited_data_size 2000 limited_task_language {t}-{l} tasks ', end='')
        for t in ts:
            print(f'{t}-{l} ', end='')
        print('&& ', end='')
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} limited_data_size 200 limited_task_language {t}-{l} tasks ', end='')
        for t in ts:
            print(f'{t}-{l} ', end='')
        print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
08.02.2019 deepnet2 run
'''
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} tasks ', end='')
        for l in ls:
            print(f'{t}-{l} ', end='')
        print('&& ', end='')
for t in ['dep', 'ner', 'pos']:
    for l in ['cs', 'es']:
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} focus_rate 0.75 tasks ', end='')
        for l in ls:
            print(f'{t}-{l} ', end='')
        print('&& ', end='')
exit()

'''
07.02.2019 martak run
'''
for t in ['dep']:
    print(f'bash train.sh adversarial_lambda 0.01 task_layer_sharing false tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
for t in ['dep']:
    print(f'bash train.sh private_params true adversarial_lambda 0.25 task_layer_sharing false tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')


'''
06.02.2019 deepnet run
'''

for t in ts:
    print(f'bash train.sh task_layer_sharing false private_params true tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
for l in ls:
    print(f'bash train.sh task_layer_sharing false private_params true tasks ', end='')
    for t in ts:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
exit()

'''
02.02.2019 martak run
'''
for t in ts:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
exit()

'''
26.01.2019 gcp run
'''
for t in ['dep']:
    print(f'bash train.sh task_layer_sharing false adversarial_lambda 0.25 tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
for t in ['dep']:
    print(f'bash train.sh task_layer_sharing false adversarial_lambda 0.125 tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
for t in ['dep']:
    print(f'bash train.sh task_layer_sharing false adversarial_frequency 2 tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
25.01.2019 martak run
'''
for t in ts:
    print(f'bash train.sh task_layer_sharing false adversarial_training false tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
print(f'bash train.sh task_layer_sharing false adversarial_training false  tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
exit()


'''
27.12.2018 gcp run
'''

for t in ts:
    for l in ls:
        print(f'bash train.sh word_lstm_size 300 tasks {t}-{l} && ', end='')

for t in ts:
    print(f'bash train.sh word_lstm_size 300 tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')

for l in ls:
    print(f'bash train.sh word_lstm_size 300 tasks ', end='')
    for t in ts:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')

print(f'bash train.sh word_lstm_size 300 tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
print('; sudo poweroff', end='')
exit()

'''
27.12.2018 gcp run
'''

for t in ts:
    for l in ls:
        print(f'bash train.sh word_lstm_size 400 tasks {t}-{l} && ', end='')

for t in ts:
    print(f'bash train.sh word_lstm_size 400 tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')

for l in ls:
    print(f'bash train.sh word_lstm_size 400 tasks ', end='')
    for t in ts:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')

print(f'bash train.sh word_lstm_size 400 tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
print('&', end='')
exit()

'''
21.12.2018 gcp run
'''
for t in ts:
    print(f'bash train.sh adversarial_training false word_emb_type mwe_projected tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
print(f'bash train.sh adversarial_training false word_emb_type mwe_projected tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
print('; sudo poweroff', end='')
exit()


'''
21.12.2018 martak run
'''
for t in ts:
    print(f'bash train.sh word_emb_type mwe_projected tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
print(f'bash train.sh word_emb_type mwe_projected tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
print('; sudo poweroff', end='')
exit()

'''
17.12.2018 martak run
'''

print(f'bash train.sh adversarial_training false tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
print('&& ', end='')

print(f'bash train.sh task_layer_sharing false tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
print('; sudo poweroff', end='')


exit()

'''
17.12.2018 gcp run
'''

for t in ts:
    print(f'bash train.sh task_layer_sharing false tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
print('; sudo poweroff')
exit()

'''
13.12.2018 gcp run
'''

for t in ts:
    print(f'bash train.sh adversarial_training false tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
print('; sudo poweroff')
exit()

'''
11.12.2018 martak run
'''

for t in ts:
    for l in ls:
        print(f'bash train.sh tasks {t}-{l} && ', end='')

for t in ts:
    print(f'bash train.sh tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')

for l in ls:
    print(f'bash train.sh tasks ', end='')
    for t in ts:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')

print(f'bash train.sh tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
print('&& ', end='')

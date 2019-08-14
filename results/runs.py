import itertools

ls = ['cs', 'de', 'en', 'es']
ts = ['dep', 'lmo', 'ner', 'pos']

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

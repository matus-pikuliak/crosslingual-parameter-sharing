import itertools

ls = ['cs', 'de', 'en', 'es']
ts = ['dep', 'lmo', 'ner', 'pos']

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
        print(f'bash train.sh private_params true task_layer_sharing false adversarial_training false focus_on {t}-{l} limited_data_size 200 limited_task_language {t}-{l}tasks ', end='')
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

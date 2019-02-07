ls = ['cs', 'de', 'en', 'es']
ts = ['dep', 'lmo', 'ner', 'pos']

'''
07.02.2019 martak run
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
print(f'bash train.sh task_layer_sharing false private_params true tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
exit()

'''
04.02.2019 gcp run
'''

for l in ls:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ', end='')
    for t in ts:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
print('; sudo poweroff', end='')
exit()

'''
02.02.2019 martak run
'''
for t in ts:
    print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ', end='')
    for l in ls:
        print(f'{t}-{l} ', end='')
    print('&& ', end='')
print(f'bash train.sh task_layer_sharing false adversarial_training false private_params true tasks ', end='')
for l in ls:
    for t in ts:
        print(f'{t}-{l} ', end='')
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
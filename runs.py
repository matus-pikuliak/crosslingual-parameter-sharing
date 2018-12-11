'''
11.12.2018 martak run
'''

ls = ['cs', 'de', 'en', 'es']
ts = ['dep', 'lmo', 'ner', 'pos']

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

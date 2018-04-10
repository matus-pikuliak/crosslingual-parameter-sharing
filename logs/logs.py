import pprint

logs = []

for i in [1,2,3,4]:
    with open('.logs//logs_0'+str(i)+'.txt') as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith("end of epoch"):
                epoch = l.split(' ')[3].strip()
            if l.startswith("acc"):
                words = l.split(' ')
                acc = words[1]
                loss = words[3]
                task = words[5]
                role = words[7]
                lang = words[9].strip()
                if role == 'dev':
                    logs += [{'file': i, 'epoch': epoch, 'task': task, 'lang': lang, 'acc': acc, 'loss': loss}]
max = {}
for log in logs:
    k = log['task']+log['lang']+str(log['file'])
    if k not in max: max[k] = log
    elif log['acc'] > max[k]['acc']:
        max[k] = log
pprint.pprint(max)
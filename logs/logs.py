import pprint

logs = []

with open('../logs.txt') as f:
    lines = f.readlines()
    for l in lines:
        if l.startswith("Now training"):
            c = len(l.split(' '))-2
            i = c/2
        if l.startswith("end of epoch"):
            epoch = l.split(' ')[3].strip()
        if l.startswith("acc"):
            words = l.split(' ')
            acc = words[1]
            loss = words[3]
            task = words[5]
            if task == 'ner':
                role = words[13]
                lang = words[7]
                recall = float(words[9])
                precision = float(words[11])
                f1 = 2 * precision * recall / (precision + recall + 0.001)
                if role == 'dev':
                    logs += [{'file': i, 'epoch': epoch, 'task': task, 'lang': lang, 'acc': acc, 'loss': loss, 'f1': f1, 'score': f1}]
            if task == 'pos':
                role = words[7]
                lang = words[9].strip()
                if role == 'dev':
                    logs += [{'file': i, 'epoch': epoch, 'task': task, 'lang': lang, 'acc': acc, 'loss': loss, 'score': acc}]
max = {}
for log in logs:
    k = log['task']+log['lang']+str(log['file'])
    if k not in max: max[k] = log
    elif log['score'] > max[k]['score']:
        max[k] = log
pprint.pprint(max)
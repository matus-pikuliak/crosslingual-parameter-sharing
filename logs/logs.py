# nacitaj vsetky  riadky zo suborov
# vytvor tabulku, s vhodnymi stlpcami
# tabulku napln
from paths import db_config
from pg import DB
db = DB(dbname=db_config['db_name'], host='localhost', port=5432, user=db_config['db_user'], passwd=db_config['db_passwd'])
db.query("create table fruits(id serial primary key, name varchar)")
#db.query("drop table fruits")
exit()



def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def is_float(str):
    try:
        float(str)
        return not is_int(str)
    except ValueError:
        return False


def is_str(str):
    return not (is_float(str) or is_int(str))

import glob
files = glob.glob('/media/fiit/5016BD1B16BD0350/Users/PC/FIIT Google Drive/data/cll-para-sharing/logs/60_epoch/*')
records = []
for file in files:
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                records.append([i.split(': ') for i in line.split(', ')])
columns = dict()
for _ in records:
    for r in _:
        c_name = r[0]
        c_value = r[1]
        if c_name not in columns:
            if is_int(c_value): columns[c_name] = 'INTEGER'
            if is_float(c_value): columns[c_name] = 'FLOAT'
            if is_str(c_value): columns[c_name] = 'VARCHAR[100]'

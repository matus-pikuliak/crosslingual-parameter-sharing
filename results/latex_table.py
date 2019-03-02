st = '''
baseline 89.16 79.93 84.92 86.51 78.10 81.36 84.96 86.77 98.89 94.19 95.99 96.46
best 89.60 79.94 84.94 86.87 81.58 83.07 85.07 87.69 99.10 94.73 96.16 96.96
fine-tuning ...
error-reduction ...
sota 91.68 80.36 84.57 90.93 - - - ? 99.07 94.50 95.94 98.80
'''

st = '''
baseline 89.16 86.51 78.10 86.77 98.89 96.46
bestunfocused 89.16 86.70 81.58 87.69 99.04 96.87
private-focusedml	89.04 86.87 78.24 86.61 98.96 96.69 
private-focusedmt	88.44 86.48 77.40 86.46 99.04 96.72 
no-adv-tsh-focusedml	88.95 86.66 77.91 87.10 99.01 96.71 
no-adv-tsh-focusedmt	87.21 85.93 78.95 86.52 99.05 96.96 
private-focused-0.75ml		89.60 86.62 78.00 86.18 99.05 96.71 
private-focused-0.75mt		88.90 86.69 77.82 86.27 99.05 96.53 
no-adv-tsh-focused-0.75ml		89.47 86.60 78.06 87.35 99.00 96.65 
no-adv-tsh-focused-0.75mt		88.44 86.68 78.12 86.83 99.10 96.48 
private-focused-0.75mtml		89.39 86.57 77.45 86.79 99.04 96.75 
no-adv-tsh-focusedmtml		87.29 86.18 80.03 86.24 98.94 96.66 

 '''

lines = st.split('\n')
r_lines = {}
for line in lines:
    if line.strip():
        split = line.split()
        r_lines[split[0]] = [float(val) for val in split[1:]]

def min_max_column(column_id):
    vals = [line[column_id] for line in r_lines.values()]
    vals = sorted(vals)[:]
    return vals[1], vals[-2]

# print(min_max_column(1))

for name, values in r_lines.items():
    print(f'{name}', end='')
    for i, val in enumerate(values):
        min_, max_ = min_max_column(i)
        shade = int((val - min_)*25 / (max_ - min_))
        if val > max_:
            print(f' & \\cellcolor{{black!{shade}}}\\textbf{{{val}}}', end='')
        else:
            print(f' & \\cellcolor{{black!{shade}}}{val}', end='')
    print('\\\\ \hline')

# for i in range(len(r_lines['Baseline'])):
#     for vals in r_lines.values():
#         print(vals[i])
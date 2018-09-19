import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import itertools

sizes = [50, 500, 1500, 5000, 15000, 67852]
pos_data = [
    [98.9161, 98.2117, 96.8601, 95.0602, 92.0322, 77.0031],
    [98.9138, 98.2198, 97.1353, 95.6708, 93.7834, 77.7811],
    [98.9161, 98.2841, 97.0936, 95.8139, 94.2278, 85.3232],
    [98.8761, 98.3217, 97.2263, 96.0723, 94.4415, 86.1145]
]

dep_uas_data = [
    [0.913995562482, 0.882063967466, 0.834276643938, 0.780035801389, 0.722823095951, 0.472810061465],
    [0.901111696858, 0.879294863254, 0.838592507198, 0.782497871034, 0.713432472671, 0.482443995157],
    [0.910427009773, 0.887468934474, 0.852797200772, 0.812871120792, 0.775395524247, 0.635080726919],
    [0.897989213238, 0.879851001338, 0.846957750885, 0.811799396358, 0.764041038356, 0.622909413216]
]

dep_las_data = [
    [0.887393624109, 0.849396648109, 0.795387529762, 0.725812338155, 0.65706556057, 0.364403686732],
    [0.871532102492, 0.844704233022, 0.796297047254, 0.728743649309, 0.650733696754, 0.378011690486],
    [0.883187829845, 0.855027546214, 0.813062293259, 0.763293727805, 0.714023369386, 0.544163736321],
    [0.866700652883, 0.844362439824, 0.806226429304, 0.756759105313, 0.70143495212, 0.528701938952]
]

ner_data = [
    [0.79256830601, 0.788990825688, 0.76105810928, 0.706218413535, 0.599450045829, 0.275713516424,],
    [0.784227820372, 0.781921895568, 0.743109151047, 0.683134965192, 0.597549909255, 0.208937198067,],
    [0.792156862745, 0.800436205016, 0.769298053794, 0.742254449571, 0.671911152244, 0.493827160493,],
    [0.786257882148, 0.768830041241, 0.758787346221, 0.723337795626, 0.648364218714, 0.50390526581],
]

# for l in dep_las_data:
#     l.reverse()
#     print ' & '.join([('%.2f' % (o*100)) for o in l])

data = [pos_data, ner_data, dep_uas_data, dep_las_data]

fig, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
plt.subplots_adjust(hspace=0.5, wspace=0.4)
ax_lst = list(itertools.chain.from_iterable(ax_lst))
y_labels = ['accuracy [%]', 'F1 score', 'unlabeled attachment score', 'labeled attachment score']
regimes = ['Baseline', 'MT', 'ML', 'MTML']
names = ['Part-of-speech tagging', 'Named entity recognition', 'Dependency parsing', 'Dependency parsing']

for ax, dat, ylabel, name in zip(ax_lst, data, y_labels, names):

    for x, reg in zip(dat, regimes):
        x.reverse()
        x = np.array(x)[1:]
        if name[0] == 'D':
            x *= 100
        y = np.array(sizes)[1:]
        ax.plot(y, x, label=reg)

    ax.set_xscale('log')
    ax.set_xticks(sizes[1:], minor=False)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.grid(True, which='major')
    ax.set_xlabel('# training samples (log scale)')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(name)


plt.show()

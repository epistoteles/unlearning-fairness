import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import matplotlib

labels = ['random', 'white', 'black', 'asian', 'indian', 'other']
dicts_monolith = pickle.load(open('summaries/R-random-balanced-slice-mean.pickle', 'rb'))
dicts_random = pickle.load(open('summaries/RX-random-balanced-shard-mean.pickle', 'rb'))
dicts_shard = pickle.load(open('summaries/R-sorted-balanced-shard-mean.pickle', 'rb'))
dicts_slice = pickle.load(open('summaries/R-sorted-balanced-slice-mean.pickle', 'rb'))

values = list(map(lambda x: x[0].reindex(index=labels).tolist(), [dicts_monolith, dicts_random, dicts_shard, dicts_slice]))
legends = ['baseline', 'uniform', 'few shards', 'later slices']

loc = np.arange(len(labels))  # the label locations
width = 0.21  # the width of the bars

sns.set_style(style='white')

upper_rects = []
fig, ax = plt.subplots(figsize=(9.5, 7), frameon=False)
for idx, (v, l, c) in enumerate(zip(values, legends, ['#002a5c', '#005f90', '#00949c', '#16c580'])):
    rect = ax.bar(loc + width * idx - width * 1.5, v, width, label=l, color=c)
    upper_rects += [rect]

lower_rects = []
# values = list(map(lambda x: np.array(x) / np.array(dicts_monolith[0][:-1].tolist()) - np.ones_like(x), values))  # scaled by monolith
values = list(map(lambda x: np.array(x) - np.array(dicts_monolith[0].reindex(index=labels).tolist()), values))  # absolute decrease
# values = [np.array(x) / np.tile(np.array(x[1]), 6) - np.ones_like(x) for x in values]  # normalized by white
for idx, (v, l) in enumerate(zip(values, legends)):
    rect = ax.bar(loc + width * idx - width * 1.5, v, width, color='darkgrey')
    lower_rects += [rect]

plt.xlim(xmin=-0.5, xmax=5.5)
plt.ylim(ymin=-0.24, ymax=1.0)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('accuracy in %', fontdict={'size': 12, 'color': '#002a5c'})
# ax.set_title('Accuracies by method and race, monolith performance=100%')
ax.set_xticks(loc)
ax.set_xticklabels(labels, fontdict={'size': 12, 'color': '#002a5c'})
ax.tick_params(axis='y', colors='#002a5c')
leg = ax.legend(legends+['decrease from\nbaseline performance'], loc='upper right', frameon=False, ncol=2)
for text in leg.get_texts():
    text.set_color('#002a5c')

def autolabel(rects, position):
    """Attach a text label above each bar in *rects*, displaying its height."""
    upper = position == 'upper'
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height*100:.1f}' if height != 0 else '',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(6, 3) if upper else (-4, -3),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom' if upper else 'top',
                    fontsize=10, rotation=60 if upper else 60,
                    color='#002a5c')


for _rect in upper_rects:
    autolabel(_rect, position='upper')
for _rect in lower_rects:
    autolabel(_rect, position='lower')

plt.rcParams["savefig.bbox"] = 'tight'
sns.despine(left=True, bottom=True, right=True)
matplotlib.rcParams['savefig.transparent'] = True
plt.savefig('plots/results-barchart.png', dpi=1200)

plt.show()

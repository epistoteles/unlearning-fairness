import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib

age_bins = [2, 9, 20, 27, 45, 65, 120]
bin_names = [*[f'{[-1, *age_bins][i] + 1}-{age_bins[i]}' for i in range(len(age_bins) - 1)], f'{age_bins[-2]}+']
races = ['white', 'black', 'asian', 'indian', 'other', 'random']


def plot_matrix(matrix, xlabels, ylabels, name):
    plt.figure(1, figsize=(12, 6))
    ax = sns.heatmap(matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".1%", annot_kws={"size": 20})
    ax.set_xticklabels(xlabels, fontdict=dict(fontsize=16))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_yticklabels(ylabels, rotation=90, va="center", fontdict=dict(fontsize=16, color='#002a5c'))
    ax.set_xlabel("class", labelpad=18, fontdict=dict(weight='bold', fontsize=16, color='#002a5c'))
    ax.set_ylabel("race", labelpad=15, fontdict=dict(weight='bold', fontsize=16, color='#002a5c'))
    colorax = plt.gcf().axes[-1]
    colorax.tick_params(length=0)
    plt.rcParams["savefig.bbox"] = 'tight'
    matplotlib.rcParams['savefig.transparent'] = True
    plt.savefig(f'plots/age-accs-{name}.png', dpi=1200)
    plt.clf()
    plt.cla()


for name in ['RXX-sorted-balanced-shard', 'RXX-sorted-balanced-slice',
             'RX-random-balanced-shard', 'R-random-balanced-slice',
             'R-sorted-balanced-slice', 'R-sorted-balanced-shard']:
    data = pickle.load(open(f'summaries/{name}-square.pickle', 'rb'))
    plot_matrix(data, bin_names, races, name)

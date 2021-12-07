from data.UTKFaceDataset import UTKFaceDataset
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

dataset = UTKFaceDataset(split='all')

plt.rcParams["savefig.bbox"] = 'tight'
f = plt.figure(figsize=(7, 3.5), frameon=False)
ax = f.add_subplot(1, 1, 1)

age_bins = [2, 9, 20, 27, 45, 65, 120]
bin_names = [*[f'{[-1, *age_bins][i] + 1}-{age_bins[i]}' for i in range(len(age_bins) - 1)], f'{age_bins[-2]}+']
races = ['white (42.5%)', 'black (19.1%)', 'asian (14.5%)', 'indian (16.8%)', 'other (7.1%)']
df = pd.DataFrame({
    "class": [bin_names[x.age_bin] for x in dataset.faces],
    "race": [races[x.race] for x in dataset.faces]
})
df['class'] = pd.Categorical(df['class'], bin_names)
df['race'] = pd.Categorical(df['race'], races[::-1])

sns.set(style="darkgrid")
sns.set_theme()
sns.histplot(data=df, ax=ax, stat='count', multiple="stack",
             x="class", kde=False, hue="race", palette="hls",
             element="bars", legend=True, binwidth=0.9, discrete=True)
leg = plt.legend(races, loc='upper left', frameon=False, prop={'size': 9})
plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    right=False,
    left=True,
    colors='#002a5c')
ax.set_xlabel("class", labelpad=10, fontdict={'color': '#002a5c'})
ax.set_ylabel("count", labelpad=10, fontdict={'color': '#002a5c'})
sns.despine(left=True, bottom=True, right=True)
matplotlib.rcParams['savefig.transparent'] = True

for text in leg.get_texts():
    text.set_color('#002a5c')

plt.savefig('plots/age-bins.png', dpi=1200)
plt.show()

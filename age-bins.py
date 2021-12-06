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
races = ['white', 'black', 'asian', 'indian', 'other']
df = pd.DataFrame({
    "class": [bin_names[x.age_bin] for x in dataset.faces],
    "race": [races[x.race] for x in dataset.faces]
})
df['class'] = pd.Categorical(df['class'], bin_names)
df['race'] = pd.Categorical(df['race'], races[::-1])

sns.set(style="darkgrid")
sns.set_theme()
sns.histplot(data=df, ax=ax, stat="percent", multiple="stack",
             x="class", kde=False, hue="race", palette="hls",
             element="bars", legend=True, binwidth=0.9, discrete=True)
plt.legend(races, loc='upper left', frameon=False)
plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    right=False,
    left=True)
ax.set_xlabel("Class")
ax.set_ylabel("Percent")
sns.despine(left=True, bottom=True, right=True)
matplotlib.rcParams['savefig.transparent'] = True
plt.savefig('plots/age-bins.png')

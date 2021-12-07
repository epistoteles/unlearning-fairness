import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

pd = [0.42901500776263585, 0.16888045540796964, 0.15094014145247542, 0.18130067276177333, 0.06986372261514577]
pdn = [0.4238244163967385, 0.19809002568971296, 0.14291299005919803, 0.16329721880933765, 0.07187534904501285]

props_did = [sum(pd[:i + 1]) for i in range(len(pd))][::-1]
props_didnt = [sum(pdn[:i + 1]) for i in range(len(pdn))][::-1]

colors = ['#d37de0', '#7d9fe0', '#7de0ab', '#c7e07d', '#e0837d'][::-1]
upper_rects = []
fig, ax = plt.subplots(figsize=(9.5, 7), frameon=False)
for idx, (v, c) in enumerate(zip(props_did, colors)):
    ax.barh(0, v, 0.9, color=c)
for idx, (v, c) in enumerate(zip(props_didnt, colors)):
    ax.barh(1, v, 0.9, color=c)

for s, k in zip((props_did[1:]+[0])[::-1], pd):
    ax.annotate(f'{k*100:.1f}%',
                xy=(s + 0.5 * k, 0),
                xytext=(0, 0),  # vertical offset
                textcoords="offset points",
                ha='center', va='center',
                fontsize=12, rotation=0,
                color='#002a5c')

for s, k in zip((props_didnt[1:]+[0])[::-1], pdn):
    ax.annotate(f'{k*100:.1f}%',
                xy=(s + 0.5 * k, 1),
                xytext=(0, 0),  # vertical offset
                textcoords="offset points",
                ha='center', va='center',
                fontsize=12, rotation=0,
                color='#002a5c')

matplotlib.rcParams['savefig.transparent'] = True
plt.savefig('plots/slice-composition.png', dpi=1200)
plt.show()

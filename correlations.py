import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, pearsonr
from data.UTKFaceDataset import UTKFaceDataset
from collections import Counter

# d = UTKFaceDataset(split='all')
# counts = [(x.race, x.age_bin) for x in d.faces]
# c = Counter(counts)
# race_counts = [[x[1] for x in sorted(c.items(), key=lambda x: x[0][1]) if x[0][0] == i] for i in range(5)]
# race_chis_uniform = [chisquare(x)[0] for x in race_counts]  # tests for uniform distribution
# race_chis = [chisquare(x, [])[0] for x in race_counts]  # tests for uniform distribution
race_chis_uniform = [3451.5319571258437, 5304.196199734866, 1723.8951659871868, 2642.859874213836, 913.0839243498817]
race_chis = []

race_proportions = [0.42509386997426485, 0.19094629371809477, 0.14487617601147534, 0.1677002911023921, 0.07138336919377294]
race_likelihoods = [0.4, 0.3, 0.43, 0.43, 0.32]

labels = ['white', 'black', 'asian', 'indian', 'other']
dicts_monolith = pickle.load(open('summaries/R-random-balanced-slice-mean.pickle', 'rb'))
dicts_random = pickle.load(open('summaries/RX-random-balanced-shard-mean.pickle', 'rb'))
dicts_shard = pickle.load(open('summaries/R-sorted-balanced-shard-mean.pickle', 'rb'))
dicts_slice = pickle.load(open('summaries/R-sorted-balanced-slice-mean.pickle', 'rb'))

values = list(map(lambda x: x[0].reindex(index=labels).tolist(), [dicts_monolith, dicts_random, dicts_shard, dicts_slice]))

# plt.figure(figsize=(6, 6))
# plt.scatter(values[2:], np.tile(np.array(race_likelihoods)*np.array(race_proportions), 2))
# plt.show()
# plt.clf()
#
# plt.figure(figsize=(6, 6))
# plt.scatter(values[2:], np.tile(np.array(race_likelihoods), 2))
# plt.show()
# plt.clf()

np.ones_like(race_chis) / np.array(race_chis)
print(values)
values1 = values  # absolute percentages
values2 = list(map(lambda x: np.array(x) / np.array(dicts_monolith[0].tolist())[:-2] - np.ones_like(x), values))  # decrease normalized by monolith
values3 = list(map(lambda x: np.array(x) - np.array(dicts_monolith[0].reindex(index=labels).tolist()), values))  # absolute decrease
values4 = [np.array(x) / np.tile(np.array(x[0]), 5) - np.ones_like(x) for x in values]  # decrease normalized by white
plt.figure(figsize=(6, 6))
for v in [values1, values2, values3, values4]:
    x, y = (np.array(v[2:]).flatten(), np.tile(np.array(race_chis), 2))
    plt.scatter(x, y)
    plt.annotate(f'{pearsonr(x, y)[0]:.3f}', xy=(x.mean(), y.mean()))
    plt.show()

import random

import torch


def random_split(dataset, lengths, random_seed):
    return list(map(list, torch.utils.data.random_split(dataset,
                                                        lengths)))  #,
                                                        #generator=torch.Generator().manual_seed(random_seed))))


def balanced_split(filenames, lengths, random_seed, test_samples):
    if lengths[-1] != 5 * 7 * test_samples:
        raise ValueError(f'False length of test set: {lengths[-1]}')
    random.seed(random_seed)
    indices = range(len(filenames))
    test_indices = []
    for race in range(5):
        for age_bin in range(7):
            candidates = list(filter(lambda x: int(x[1].split('_')[2]) == race,
                                     zip(indices, filenames)))
            candidates = list(filter(lambda x: next(z[0] for z in enumerate([2, 9, 20, 27, 45, 65, 120])
                                                    if z[1] >= int(x[1].split('_')[0])) == age_bin,
                                     candidates))
            if len(candidates) < test_samples:
                raise ValueError(f'Not enough samples with race {race} and age bin {age_bin}: {len(candidates)} < {test_samples}')
            selected = random.sample(candidates, test_samples)
            print(selected)
            selected_indices = list(map(lambda x: x[0], selected))
            test_indices += selected_indices
    remaining_indices = [x for x in indices if x not in test_indices]
    return random_split(remaining_indices, lengths[:-1], random_seed=42) + [test_indices]


def split(a, n):
    k, m = divmod(len(a), n)  # len(a)=43, n=4, k=10, m=3
    return [k + int(i < m) for i in range(n)]


def get_counts(filenames, num_shards, num_slices, test=0.1, return_indices=False, strategy='random'):
    length = len(filenames)
    test_samples = 9
    if type(num_shards) is tuple:
        num_shards = num_shards[0]
    if type(num_slices) is tuple:
        num_slices = num_slices[0]
    if strategy == random:
        test_len = int(length * test)
    else:
        test_len = 5 * 7 * test_samples  # races * age_bins * test_samples
    train_val_len = length - test_len
    n = num_shards * num_slices
    d, m = divmod(train_val_len, n)
    lengths = [*[d + int(i < m) for i in range(n)],
               test_len]  # [shard_1_slice_1, shard_1_slice_2, shard_2_slice_1, ..., test]
    if not return_indices:
        return lengths
    elif strategy == 'random':
        return random_split(range(length), lengths, random_seed=42)
    elif strategy == 'balanced':
        return balanced_split(filenames, lengths, random_seed=42, test_samples=test_samples)


def random_run_name():
    animals = ['bear', 'tiger', 'panther', 'scorpion', 'owl', 'salmon']
    adjectives = ['lazy', 'happy', 'green', 'blue', 'hungry', 'warm', 'royal', 'bored', 'marble', 'striped',
                  'brown', 'toxic', 'siberian', 'musical']
    return f"{random.choice(adjectives)}-{random.choice(animals)}"



import random
import torch


def random_split(dataset, lengths, random_seed):
    return list(map(list, torch.utils.data.random_split(dataset,
                                                        lengths,
                                                        generator=torch.Generator().manual_seed(random_seed))))


def sequential_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError(f"Lengths don't match: dataset has length {len(dataset)}, but {sum(lengths)} was requested")
    result = []
    for length in lengths:
        result += [dataset[:length]]
        dataset = dataset[length:]
    return result

def balanced_split(faces, lengths, random_seed, num_test_samples, train, put_in='shard', num_slices=1, num_shards=1):
    if lengths[-1] != 5 * 7 * num_test_samples:
        raise ValueError(f'False length of test set: {lengths[-1]}')
    random.seed(random_seed)
    indices = range(len(faces))
    test_indices = []
    for race in range(5):
        for age_bin in range(7):
            candidates = list(filter(lambda x: x[1].race == race, zip(indices, faces)))
            candidates = list(filter(lambda x: x[1].age_bin == age_bin, candidates))
            if len(candidates) < num_test_samples:
                raise ValueError(f'Not enough samples with race {race} and age bin {age_bin}: {len(candidates)} < {num_test_samples}')
            selected = random.sample(candidates, num_test_samples)
            selected_indices = list(map(lambda x: x[0], selected))
            test_indices += selected_indices
    remaining_indices = [x for x in indices if x not in test_indices]
    if train == 'random':
        return random_split(remaining_indices, lengths[:-1], random_seed=42) + [test_indices]
    elif train == 'sorted':
        high_risk_indices = [x for x in remaining_indices if faces[x].changed_privacy_settings]
        low_risk_indices = [x for x in remaining_indices if not faces[x].changed_privacy_settings]
        random.shuffle(high_risk_indices)
        random.shuffle(low_risk_indices)
        all_indices = high_risk_indices + low_risk_indices
        train_indices = sequential_split(all_indices, lengths[:-1])
        if put_in == 'slice':
            indices = list(range(len(train_indices)))
            new_order = [0]*len(train_indices)
            for i in reversed(range(num_slices)):
                for j in range(num_shards):
                    new_order[num_slices*j+i] = indices.pop(0)
            train_indices = [train_indices[i] for i in new_order]
        return train_indices + [test_indices]


def split(a, n):
    k, m = divmod(len(a), n)  # len(a)=43, n=4, k=10, m=3
    return [k + int(i < m) for i in range(n)]


def get_indices(faces, num_shards, num_slices, test=0.1, strategy='random-random', random_seed=42, put_in='shard'):
    length = len(faces)
    num_test_samples = 9
    if type(num_shards) is tuple:
        num_shards = num_shards[0]
    if type(num_slices) is tuple:
        num_slices = num_slices[0]
    if strategy == random:
        test_len = int(length * test)
    else:
        test_len = 5 * 7 * num_test_samples  # races * age_bins * test_samples
    train_val_len = length - test_len
    n = num_shards * num_slices
    d, m = divmod(train_val_len, n)
    lengths = [*[d + int(i < m) for i in range(n)],
               test_len]  # [shard_0_slice_0, shard_0_slice_1, shard_1_slice_0, ..., test]
    if strategy == 'random-random':
        return random_split(range(length), lengths, random_seed=random_seed)
    elif strategy == 'random-balanced':
        return balanced_split(faces, lengths,
                              random_seed=random_seed,
                              num_test_samples=num_test_samples,
                              train='random')
    elif strategy == 'sorted-balanced':
        return balanced_split(faces, lengths,
                              random_seed=random_seed,
                              num_test_samples=num_test_samples,
                              train='sorted',
                              put_in=put_in)



def random_run_name():
    animals = ['bear', 'tiger', 'panther', 'scorpion', 'owl', 'salmon']
    adjectives = ['lazy', 'happy', 'green', 'blue', 'hungry', 'warm', 'royal', 'bored', 'marble', 'striped',
                  'brown', 'toxic', 'siberian', 'musical']
    return f"{random.choice(adjectives)}-{random.choice(animals)}"



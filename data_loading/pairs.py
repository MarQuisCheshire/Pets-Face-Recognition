import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .dataset import RecDataset


class PairGenerator(Dataset):
    dataset: RecDataset

    def __init__(self, dataset, gen_number=None, gen_ratio=1, path=None, random_seed=None, usr_list=None):
        self.dataset = dataset
        if path is None or not Path(path).exists():
            self.generate_pairs(gen_number, gen_ratio, path, random_seed, usr_list)
        else:
            with open(path, 'rb') as f:
                self.pairs, self.correction = pickle.load(f)

    def __getitem__(self, item):
        indices = self.pairs[item]
        x1 = self.dataset[indices[0]]['x']
        x2 = self.dataset[indices[1]]['x']

        return {'x1': x1, 'x2': x2, 'label': int(indices[2])}

    def __len__(self):
        return len(self.pairs)

    def generate_pairs(self, gen_number, gen_ratio, path, random_seed, usr_list):
        rand = np.random.RandomState(random_seed)
        l = len(self.dataset)

        usr_list = set(usr_list)

        max_gen_number = sum(len(i) * len(i) - len(i) for u, i in self.dataset.uid_to_indices.items() if u in usr_list)
        max_imp_number = sum(
            l * len(i) - min(l, len(i)) for u, i in self.dataset.uid_to_indices.items() if u in usr_list
        )
        if gen_number is not None:
            assert gen_number <= max_gen_number, f'{gen_number} greater than {max_gen_number}'
        else:
            gen_number = max_gen_number
        imp_number = int(gen_number * gen_ratio)
        assert imp_number <= max_imp_number, f'{imp_number} greater than {max_imp_number}'

        # positive pairs
        gen_pairs = []
        gen_parts = {
            user: len(i) * len(i) - len(i) for user, i in self.dataset.uid_to_indices.items()
            if user in usr_list and len(i) > 1
        }
        for i, j in gen_parts.items():
            n = min(round(j / max_gen_number * gen_number), j)
            pairs = [
                (ii, jj) for ii in self.dataset.uid_to_indices[i] for jj in self.dataset.uid_to_indices[i] if ii != jj
            ]
            gen_pairs.extend([pairs[ii] for ii in rand.choice(len(pairs), n, replace=False)])

        # negative pairs
        imp_pairs = []
        imp_parts = {
            user: l * len(i) - min(l, len(i)) for user, i in self.dataset.uid_to_indices.items()
            if user in usr_list
        }
        all_indices = {j for u, i in self.dataset.uid_to_indices.items() if u in usr_list for j in i}
        for i, j in imp_parts.items():
            n = min(round(j * imp_number / max_imp_number), j)
            tmp = all_indices - set(self.dataset.uid_to_indices[i])
            pairs = [
                (ii, jj) for ii in self.dataset.uid_to_indices[i] for jj in tmp
            ]
            imp_pairs.extend([pairs[ii] for ii in rand.choice(len(pairs), n, replace=False)])

        # correction
        correction = {i: 0 for i in all_indices}
        last_shift = 0
        previous = None
        for i in sorted(correction):
            if previous is not None:
                last_shift += i - previous - 1
                correction[i] = i - last_shift
            else:
                last_shift = i
            previous = i

        pairs = [(i[0], i[1], 1) for i in gen_pairs]
        pairs.extend((i[0], i[1], 0) for i in imp_pairs)

        if path is not None:
            with open(path, 'wb') as f:
                pickle.dump([pairs, correction], f)

        self.pairs = pairs
        self.correction = correction

    @property
    def labels(self):
        return np.array([int(i) for _, _, i in self.pairs])

    @property
    def indices(self):
        return [(i, j) for i, j, _ in self.pairs]

    @property
    def corrected_indices(self):
        return [(self.correction[i], self.correction[j]) for i, j, _ in self.pairs]


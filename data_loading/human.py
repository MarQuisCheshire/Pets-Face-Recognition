import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms
from PIL import Image
from pipe import where, select
from torch.utils.data import Dataset


def init_dataset_ms1m(path, **_):
    path = Path(path)
    user_to_paths = {}
    for dir_ in path.iterdir():
        img_paths = [i for i in dir_.iterdir()]
        user_to_paths[dir_] = img_paths

    return user_to_paths


class LFWDataset(Dataset):

    def __init__(self):
        self.keys = {}
        self.dataset = {}
        self.p = Path('lfw/lfw-deepfunneled/lfw-deepfunneled')
        c = 0
        dirs = pd.read_csv(str(Path('lfw/people.csv')))['name'].tolist() | where(lambda x: isinstance(x, str))
        for dir_ in dirs:
            if isinstance(dir_, str):
                files = tuple((self.p / dir_).iterdir())
                keys = [i.name[:-4].split('_') for i in files]
                keys = [('_'.join(i[:-1]), int(i[-1])) for i in keys]
                for key, f in zip(keys, files):
                    self.dataset[key] = f
                    self.keys[c] = key
                    c += 1

        self.inverted_keys = dict(i[::-1] for i in self.keys.items())
        self.labels = {}
        c = 0
        for i in sorted(set(self.dataset | select(lambda x: x[0]))):
            self.labels[i] = c
            c += 1
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, item):
        name, id_ = self.keys[item]
        img = np.array(Image.open(self.dataset[name, id_]))
        label = self.labels[name]
        img = self.to_tensor(img)
        return {'x': img, 'label': label, 'index': item}

    def __len__(self):
        return len(self.dataset)


# class LFWPairGenerator(Dataset):
#
#     def __init__(self, dataset):
#         self.dataset: LFWDataset = dataset
#         p = Path('lfw/matchpairsDevTest.csv')
#         gen_pairs = pd.read_csv(str(p))
#         p = Path('lfw/mismatchpairsDevTest.csv')
#         imp_pairs = pd.read_csv(str(p))
#
#         self.pairs = [(row[0], row[1], row[0], row[2]) for _, row in gen_pairs.iterrows()]
#         self.pairs.extend(tuple(row) for _, row in imp_pairs.iterrows())
#
#     def __getitem__(self, item):
#         name1, id1, name2, id2 = self.pairs[item]
#         x1 = self.dataset[self.dataset.inverted_keys[name1, id1]]['x']
#         x2 = self.dataset[self.dataset.inverted_keys[name2, id2]]['x']
#         return {'x1': x1, 'x2': x2, 'label': int(name1 == name2)}
#
#     def __len__(self):
#         return len(self.pairs)
#
#     @property
#     def labels(self):
#         return np.array([int(i[0] == i[2]) for i in self.pairs])
#
#     @property
#     def indices(self):
#         d = self.dataset.inverted_keys
#         return [(d[i, j], d[k, l]) for i, j, k, l in self.pairs]
#
#     @property
#     def corrected_indices(self):
#         return self.indices


class LFWPairGenerator(Dataset):

    def __init__(self, dataset, gen_number=None, gen_ratio=1, path=None, random_seed=None, usr_list=None):
        self.dataset: LFWDataset = dataset
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
        user_to_indices = defaultdict(list)
        for i, j in self.dataset.keys.items():
            user_to_indices[self.dataset.labels[j[0]]].append(i)

        max_gen_number = sum(len(i) * len(i) - len(i) for u, i in user_to_indices.items() if u in usr_list)
        max_imp_number = sum(
            l * len(i) - min(l, len(i)) for u, i in user_to_indices.items() if u in usr_list
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
            user: len(i) * len(i) - len(i) for user, i in user_to_indices.items() if user in usr_list and len(i) > 1
        }
        for i, j in gen_parts.items():
            n = min(round(j / max_gen_number * gen_number), j)
            pairs = [
                (ii, jj) for ii in user_to_indices[i] for jj in user_to_indices[i] if ii != jj
            ]
            gen_pairs.extend([pairs[ii] for ii in rand.choice(len(pairs), n, replace=False)])

        # negative pairs
        imp_pairs = []
        imp_parts = {
            user: l * len(i) - min(l, len(i)) for user, i in user_to_indices.items()
            if user in usr_list
        }
        all_indices = {j for u, i in user_to_indices.items() if u in usr_list for j in i}
        for i, j in imp_parts.items():
            n = min(round(j * imp_number / max_imp_number), j)
            tmp = all_indices - set(user_to_indices[i])
            pairs = [
                (ii, jj) for ii in user_to_indices[i] for jj in tmp
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

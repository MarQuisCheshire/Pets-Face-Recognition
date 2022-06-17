import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from albumentations import bbox_rot90, keypoint_rot90
from pipe import where
from torch.utils.data import Dataset
from tqdm import tqdm


def check_dir(path, type_, min_number):
    path = Path(path)
    if not path.is_dir():
        return False
    with open(path / 'card.json', 'r', encoding='utf-8') as fp:
        info = json.load(fp)
    return (
            len([i for i in path.iterdir() if i.name != 'card.json']) >= min_number and
            # info['pet']['petBreed']['type'] == type_ and
            int(info['pet']['animal']) == type_
        # int(info['pet']['type']) == type_
    )


def check(paths, preprocessor=None):
    ok = []
    for path in paths:
        try:
            img = np.asarray(Image.open(path))
            if preprocessor:
                preprocessor(img)
            ok.append(path)
        except Exception:
            pass
    return ok


def init_dataset(path, type_=1, min_number=3, preprocessor=None, paths_to_exclude=None):
    if paths_to_exclude is None:
        paths_to_exclude = set()
    else:
        paths_to_exclude = {Path(i).resolve() for i in paths_to_exclude}
    path = Path(path)
    user_to_paths = {}
    msg = 'Analyzing the data and preparing the datasets'
    for dir_ in tqdm(tuple(path.iterdir() | where(lambda x: check_dir(x, type_, min_number))), desc=msg):
        img_paths = [i for i in dir_.iterdir() if i.name != 'card.json' and i.resolve() not in paths_to_exclude]
        img_paths = check(img_paths, preprocessor)
        if len(img_paths) >= min_number:
            user_to_paths[dir_] = img_paths

    return user_to_paths


def simple_init_dataset(path, type_, min_number, *_, **__):
    path = Path(path)
    user_to_paths = {}
    for dir_ in path.iterdir():
        img_paths = list(dir_.iterdir())
        if len(img_paths) >= min_number:
            user_to_paths[dir_] = img_paths
    return user_to_paths


class RecDataset(Dataset):

    def __init__(
            self,
            path,
            type_,
            min_number,
            preprocessor=None,
            train_augmentation=None,
            val_augmentation=None,
            init_dataset_method=init_dataset,
            paths_to_exclude=None,
            val_indices=None,
            start_class=0
    ):
        self.user_to_paths = init_dataset_method(path, type_, min_number, preprocessor, paths_to_exclude)
        self.preprocessor = preprocessor
        self.start_class = start_class
        self.train_augmentation = train_augmentation
        self.val_augmentation = val_augmentation
        self.uid_to_user = dict(enumerate(sorted(set(self.user_to_paths), key=lambda x: str(x.name))))
        self.user_to_uid = {j: i for i, j in self.uid_to_user.items()}
        tmp = [(i, j) for i in self.user_to_paths for j in self.user_to_paths[i]]
        tmp = sorted(tmp, key=lambda x: (str(x[0].name), str(x[1].name)))
        self.index_to_uid = {i: self.user_to_uid[j[0]] for i, j in enumerate(tmp)}
        self.index_to_path = {i: j[1] for i, j in enumerate(tmp)}
        self.uid_to_indices = defaultdict(list)
        for i, j in self.index_to_uid.items():
            self.uid_to_indices[j].append(i)
        self.uid_to_indices = dict(self.uid_to_indices)
        self.val_indices = val_indices
        self.label_map = dict(zip(self.uid_to_user.keys(), range(len(self.uid_to_user))))

    def __getitem__(self, item):
        if item < 0:
            item += len(self)
        path = self.index_to_path[item]
        if path.name[-4:] in ('.jpg', '.png', '.JPG', 'jpeg'):
            try:
                img = np.asarray(Image.open(path).convert("RGB"))
            except OSError:
                print(item, path)
        elif path.name[-4:] == '.npy':
            img = np.load(path)
        else:
            raise Exception('Unsupported file format')
        label = self.index_to_uid[item]
        resolved_label = self.label_map[label]

        if self.preprocessor:
            img = self.preprocessor(img)
        if (self.val_indices is None or item not in self.val_indices) and self.train_augmentation:
            img = self.train_augmentation(img)
        elif self.val_augmentation:
            img = self.val_augmentation(img)

        resolved_label += self.start_class

        return {'x': img, 'label': resolved_label, 'index': item}

    def __len__(self):
        return len(self.index_to_path)

    def get_users(self):
        return list(self.user_to_uid.values())

    @property
    def val_indices(self):
        return self._val_indices

    @val_indices.setter
    def val_indices(self, value):
        if value is not None:
            self._val_indices = set(value)
        else:
            self._val_indices = value


class SimpleDataset(Dataset):

    def __init__(self, root, paths, others, transform=None, rotate90=False):
        self.root = Path(root)
        self.paths = paths
        self.others = others
        self.transform = transform
        self.rotate90 = rotate90

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = np.array(Image.open(self.root / self.paths[item].replace('\\', '/')))
        others = self.others[item]
        initial_sizes = image.shape[:2]

        if self.rotate90:
            angle = np.random.randint(0, 4)
            if angle:
                image = np.ascontiguousarray(np.rot90(image, angle))
                bbox = others['boxes'].astype(float) / np.asarray(list(initial_sizes)[::-1] * 2)
                bbox = np.asarray([bbox_rot90(bbox[i], angle, 0, 0) for i in range(len(bbox))])
                others['boxes'] = np.round(bbox * np.asarray(list(image.shape[:2])[::-1] * 2)).astype(np.int64)
                others['keypoints'][:, :-1] = self.process_keypoints(
                    others['keypoints'][:, :-1],
                    angle,
                    initial_sizes,
                    keypoint_rot90
                )

        if self.transform:
            image, others = self.transform(image, others)

        return image, others

    @staticmethod
    def process_keypoints(keypoints, angle, initial_sizes, rotate_fn):
        new_keypoints = []
        for i in range(len(keypoints)):
            new_keypoints.append(rotate_fn(tuple(list(keypoints[i]) + [0, 0]), angle, *initial_sizes)[:2])
        return np.asarray(new_keypoints)


class RecSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, item):
        data = self.dataset[self.indices[item]]
        if self.transform:
            data['x'] = self.transform(data['x'])
        return data

    def __len__(self):
        return len(self.indices)

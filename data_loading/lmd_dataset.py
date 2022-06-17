from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from albumentations import rotate, bbox_rotate, keypoint_rotate, bbox_rot90, keypoint_rot90
from torch.utils.data import Dataset


class LMDDataset(Dataset):

    def __init__(self, celeba, oxford, oxford_transform=None):
        self.oxford_dataset = oxford
        self.celeba = celeba
        self.oxford_transform = oxford_transform
        self.support_indexing = np.random.permutation(
            list(range(len(self.oxford_dataset))) * (1 + len(self.celeba) // len(self.oxford_dataset))
        )[:len(self.celeba)]
        assert len(self.support_indexing) == len(self.celeba)

    def __getitem__(self, item):
        celeba_img, (celeba_bbox, celeba_lmd) = self.celeba[item]
        oxford_img, oxford_bbox = self.oxford_dataset[self.support_indexing[item]]
        celeba_lmd = celeba_lmd.float()
        oxford_bbox = oxford_bbox['boxes'][0]

        cropepd_celeba_img = celeba_img
        # cropped_oxford_img = oxford_img

        # cropepd_celeba_img = celeba_img[celeba_bbox[0]:celeba_bbox[2], celeba_bbox[1]:celeba_bbox[3]]
        cropped_oxford_img = np.array(Image.fromarray(oxford_img).crop(oxford_bbox))
        if self.oxford_transform:
            cropped_oxford_img = self.oxford_transform(cropped_oxford_img)

        for i in range(len(celeba_lmd)):
            celeba_lmd[i] = celeba_lmd[i] / cropepd_celeba_img.shape[i % 2]

        return {
            'human': cropepd_celeba_img,
            'animal': cropped_oxford_img,
            'lmd': celeba_lmd[:6]
        }

    def __len__(self):
        return len(self.celeba)


class CatLMDDataset(Dataset):

    def __init__(self, path):
        self.paths = [f_p for d in path.iterdir() for f_p in d.glob('*.jpg')]
        self.lmd = [self.read_lmd(path) for path in self.paths]

    def __getitem__(self, item):
        path = self.paths[item]
        image = np.array(Image.open(path))
        lmd = self.lmd[item]
        lmd = np.array([(lmd[i], lmd[i + 1], 1) for i in range(0, len(lmd), 2)])
        center = (lmd[0][:-1] + lmd[1][:-1]) / 2
        dif_eyes = ((lmd[0][:-1] - lmd[1][:-1]) ** 2).sum() ** 0.5
        dif_nose = ((center - lmd[2][:-1]) ** 2).sum() ** 0.5
        bbox = [
            max(0, min(center[0] - dif_eyes * 1.4, *(lmd[:, 0] - 1))),
            max(0, min(center[1] - dif_nose * 1.8, *(lmd[:, 1] - 1))),
            min(image.shape[1] - 1, max(center[0] + dif_eyes * 1.4, *(lmd[:, 0] + 1))),
            min(image.shape[0] - 1, max(center[1] + dif_nose * 1.8, *(lmd[:, 1] + 1)))
        ]
        # add = [5, 5, 5, 10]
        # bbox = [min(lmd[::2]), min(lmd[1::2]), max(lmd[::2]), max(lmd[1::2])]
        # bbox = [bbox[i] + add[i] for i in range(len(bbox))]
        bbox = np.round(np.asarray(bbox))
        lmd = lmd[:3]

        target = {'boxes': bbox, 'keypoints': lmd, 'labels': [0]}
        return image, target

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def read_lmd(path: Path):
        new_path = Path(str(path.resolve()) + '.cat')
        with open(new_path, 'r') as f:
            lines = f.readlines()
        lmd = list(map(int, lines[0].split()))[1:]
        return lmd


class CatLMDSubset(Dataset):

    def __init__(self, dataset, indices, transform=None, rotate=False, rotate90=False) -> None:
        self.dataset: CatLMDDataset = dataset
        self.indices = indices
        self.transform = transform
        assert int(bool(rotate)) + int(rotate90) < 2, 'It is not supported to use rotate and rotate90 at the same time'
        self.rotate = rotate
        self.rotate90 = rotate90

    def __getitem__(self, idx):
        image, target_d = self.dataset[self.indices[idx]]

        initial_sizes = image.shape[:2]

        if self.rotate:
            angle = np.random.uniform(-self.rotate, self.rotate)
            image = rotate(image, angle, cv2.INTER_NEAREST, cv2.BORDER_REFLECT_101, None)

            bbox = target_d['boxes'].astype(float) / np.asarray(list(initial_sizes)[::-1] * 2)
            bbox = bbox_rotate(bbox, angle, *initial_sizes)
            target_d['boxes'] = np.round(bbox * np.asarray(list(image.shape[:2])[::-1] * 2)).astype(np.int64)

            # keypoints = target_d['keypoints'][:, :-1].astype(float) / np.asarray(initial_sizes)
            target_d['keypoints'][:, :-1] = self.process_keypoints(
                target_d['keypoints'][:, :-1],
                angle,
                initial_sizes,
                keypoint_rotate
            )
            target_d['keypoints'][:, -1] = (
                ~(
                        (target_d['keypoints'][:, :-1] < 0).any(axis=-1) |
                        (target_d['keypoints'][:, :-1] > initial_sizes).any(axis=-1)
                )
            ).astype(target_d['keypoints'].dtype)
            # target_d['keypoints'][:, :-1] = np.round(keypoints * np.asarray(image.shape[:2])).astype(np.int64)
        elif self.rotate90:
            angle = np.random.randint(0, 4)

            if angle:
                image = np.ascontiguousarray(np.rot90(image, angle))
                bbox = target_d['boxes'].astype(float) / np.asarray(list(initial_sizes)[::-1] * 2)
                bbox = bbox_rot90(bbox, angle, 0, 0)
                target_d['boxes'] = np.round(bbox * np.asarray(list(image.shape[:2])[::-1] * 2)).astype(np.int64)
                target_d['keypoints'][:, :-1] = self.process_keypoints(
                    target_d['keypoints'][:, :-1],
                    angle,
                    initial_sizes,
                    keypoint_rot90
                )

        if self.transform:
            image, target = self.transform(image, target_d)

        target_d['boxes'] = [target_d['boxes']]
        # target_d['keypoints'] = target_d['keypoints'][None]
        return image, target_d

    @staticmethod
    def process_keypoints(keypoints, angle, initial_sizes, rotate_fn):
        new_keypoints = []
        for i in range(len(keypoints)):
            new_keypoints.append(rotate_fn(tuple(list(keypoints[i]) + [0, 0]), angle, *initial_sizes)[:2])
        return np.asarray(new_keypoints)

    def __len__(self):
        return len(self.indices)

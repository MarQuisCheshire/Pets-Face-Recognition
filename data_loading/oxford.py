import os
import os.path
import pathlib
import xml.etree.ElementTree as ET
from typing import Any, Callable, Optional, Union, Tuple, List
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from albumentations import rotate, bbox_rotate, bbox_rot90
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive


class OxfordIIITPet(VisionDataset):
    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "bbox", "segmentation", "body_bbox", "big_class")

    def __init__(
            self,
            root: str,
            split: Optional[Union[str, List]] = None,
            target_types: Union[Sequence[str], str] = "category",
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        self._split = split if split is not None else ("trainval", "test")
        if isinstance(target_types, str):
            target_types = [target_types]
        self.target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._bbox_folder = self._anns_folder / "xmls"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        s = {i.name[:-4] for i in self._bbox_folder.iterdir()}
        image_ids = []
        self._labels = []
        for split in self._split:
            with open(self._anns_folder / f"{split}.txt") as file:
                for line in file:
                    image_id, label, *_ = line.strip().split()
                    if image_id in s:
                        image_ids.append(image_id)
                        self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        t = [self._parse_xml(self._bbox_folder / f"{image_id}.xml") for image_id in image_ids]
        self._bbox, self.big_classes = tuple(map(list, zip(*t)))
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]
        self._body_bbox = None

        if 'body_bbox' in self.target_types:
            to_delete = set()
            self._body_bbox = {}
            for i in range(len(self._segs)):
                img = (np.array(Image.open(self._segs[i])) != 2).astype(int)
                if img.sum() != 0:
                    tmp = (np.sum(img, axis=0) == 0).tolist()
                    x1, x2 = tmp.index(False), len(tmp) - tmp[::-1].index(False)
                    tmp = (np.sum(img, axis=1) == 0).tolist()
                    y1, y2 = tmp.index(False), len(tmp) - tmp[::-1].index(False)
                    assert x1 < x2 and y1 < y2
                    self._body_bbox[len(self._body_bbox)] = (x1, y1, x2, y2)
                else:
                    to_delete.add(i)
            self._segs = [self._segs[j] for j in range(len(self._segs)) if j not in to_delete]
            self._bbox = [self._bbox[j] for j in range(len(self._bbox)) if j not in to_delete]
            self.big_classes = [self.big_classes[j] for j in range(len(self.big_classes)) if j not in to_delete]
            self._images = [self._images[j] for j in range(len(self._images)) if j not in to_delete]
            self._labels = [self._labels[j] for j in range(len(self._labels)) if j not in to_delete]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self.target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            elif target_type == 'big_class':
                target.append(self.big_classes[idx])
            elif target_type == "bbox":
                target.append([np.array(self._bbox[idx], dtype=np.int64)])
            elif target_type == 'body_bbox':
                target.append([np.array(self._body_bbox[idx], dtype=np.int64)])
            else:  # target_type == "segmentation"
                m = np.array(Image.open(self._segs[idx]))
                m = (m != 2).astype(int)
                target.append(m)

        if not target:
            target = None
        # elif len(target) == 1:
        #     target = target[0]
        else:
            target = tuple(target)

        image = np.array(image)

        return image, target

    def _parse_xml(self, path: pathlib.Path):
        d = dict(zip(('xmin', 'ymin', 'xmax', 'ymax', 'name'), [None] * 5))
        for event, elem in ET.iterparse(str(path)):
            if elem.tag in d:
                d[elem.tag] = elem.text
        assert all(i is not None for i in d.values())
        t = tuple(d.values())
        return [int(i) for i in t[:-1]], ['dog', 'cat'].index(t[-1])

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


class OxfordSubset(Dataset):

    def __init__(
            self, dataset,
            indices: Sequence[int],
            transform=None,
            rotate=False,
            rotate90=False,
            big_classes=False
    ) -> None:
        self.dataset: OxfordIIITPet = dataset
        self.indices = indices
        self.transform = transform
        assert int(bool(rotate)) + int(rotate90) < 2, 'It is not supported to use rotate and rotate90 at the same time'
        self.rotate = rotate
        self.rotate90 = rotate90
        self.big_classes = big_classes

    def __getitem__(self, idx):
        image, target_list = self.dataset[self.indices[idx]]
        if list(self.dataset.target_types) in (['bbox'], ['body_bbox'], ['bbox', 'body_bbox'], ['body_bbox', 'bbox']):
            initial_sizes = image.shape[:2]

            if self.rotate:
                angle = np.random.uniform(-self.rotate, self.rotate)
                image = rotate(image, angle, cv2.INTER_NEAREST, cv2.BORDER_REFLECT_101, None)
                for target in target_list:
                    for i in range(len(target)):
                        bbox = target[i].astype(float) / np.asarray(list(initial_sizes)[::-1] * 2)
                        bbox = bbox_rotate(bbox, angle, *initial_sizes)
                        target[i] = np.round(bbox * np.asarray(list(image.shape[:2])[::-1] * 2)).astype(np.int64)
            elif self.rotate90:
                angle = np.random.randint(0, 4)
                image = np.ascontiguousarray(np.rot90(image, angle))
                for target in target_list:
                    for i in range(len(target)):
                        bbox = target[i].astype(float) / np.asarray(list(initial_sizes)[::-1] * 2)
                        bbox = bbox_rot90(bbox, angle, *initial_sizes)
                        target[i] = np.round(bbox * np.asarray(list(image.shape[:2])[::-1] * 2)).astype(np.int64)

            if self.transform:
                image, target = self.transform(image, target_list)

            for target in target_list:
                for i in range(len(target)):
                    if isinstance(image, torch.Tensor):
                        new_sizes = image.shape[1:]
                    else:
                        new_sizes = image.shape[:2]
                    if (np.asarray(new_sizes) != np.asarray(initial_sizes)).any():
                        target[i] = target[i].astype(float) / np.asarray(list(initial_sizes)[::-1] * 2)
                        target[i] = np.round(target[i] * np.asarray(list(new_sizes)[::-1] * 2)).astype(np.int64)
            if self.big_classes:
                if list(self.dataset.target_types) in (['bbox'], ['body_bbox']):
                    labels = [self.dataset.big_classes[idx]] * len(target_list[0])
                else:
                    labels = [0] * len(target_list[0]) + [self.dataset.big_classes[idx] + 1] * len(target_list[1])
            else:
                labels = [0] * len(target_list[0])
                if len(target_list) == 2:
                    labels += [1] * len(target_list[1])

            boxes = [j for i in target_list for j in i]
            return image, {"boxes": boxes, "labels": labels}

        elif list(self.dataset.target_types) == ['segmentation']:
            target = target_list[0]
            if self.transform:
                image, target = self.transform(image, target)

            if self.big_classes:
                # target = [np.zeros(target.shape) if i != self.dataset.big_classes[idx] else target for i in range(3)]
                # target = torch.tensor(np.concatenate([i[None] for i in target]))
                target = torch.tensor(
                    (np.array(target) * (self.dataset.big_classes[idx] + 1))
                ).long()
            else:
                target = np.array(target)
                target = [np.zeros(target.shape) if i != self.dataset.big_classes[idx] + 1 else target for i in
                          range(3)]
                target = torch.tensor(np.concatenate([i[None] for i in target]))
            return image, target

        elif list(self.dataset.target_types) in (['segmentation', 'body_bbox'], ['body_bbox', 'segmentation']):
            assert not self.rotate, f'Currently not supported for these target_types {self.dataset.target_types}'
            if self.rotate90:
                target_list = list(target_list)
                initial_sizes = image.shape[:2]
                angle = np.random.randint(0, 4)
                if angle:
                    image = np.ascontiguousarray(np.rot90(image, angle))
                    segmentation = target_list[list(self.dataset.target_types).index('segmentation')]
                    segmentation = np.ascontiguousarray(np.rot90(segmentation, angle))
                    target_list[list(self.dataset.target_types).index('segmentation')] = segmentation
                    for target in target_list:
                        for i in range(len(target)):
                            coef = np.asarray(list(initial_sizes)[::-1] * 2)
                            bbox = target[i].astype(float) / coef
                            bbox = bbox_rot90(bbox, angle, 0, 0)
                            target[i] = np.round(bbox * coef[::-1]).astype(np.int64)

            if self.transform:
                image, target_list = self.transform(image, target_list)

            labels = []
            segmentation = target_list[list(self.dataset.target_types).index('segmentation')]
            boxes = target_list[list(self.dataset.target_types).index('body_bbox')]

            if self.big_classes:
                labels.append(self.dataset.big_classes[idx] + 1)
            else:
                labels.append(0)

            boxes = np.asarray(boxes)
            segmentation = torch.as_tensor(segmentation[None]).to(torch.uint8)
            return image, {"boxes": boxes, "labels": labels, "masks": segmentation}

    def __len__(self):
        return len(self.indices)

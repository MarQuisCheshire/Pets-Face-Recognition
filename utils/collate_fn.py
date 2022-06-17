from copy import deepcopy

import numpy as np
import torch


def detection_collate_fn(batch):
    img = torch.cat([i.unsqueeze(0) for i, _ in batch], dim=0)
    meta = [{k: torch.tensor(deepcopy(np.array(i[k]))).long() for k in i} for _, i in batch]
    return img, meta


def detection_collate_list_fn(batch):
    img = [i for i, _ in batch]
    meta = [{k: torch.tensor(deepcopy(np.array(i[k]))).long() for k in i} for _, i in batch]
    return img, meta


def key_points_collate_list_fn(batch):
    img = [i for i, _ in batch]
    meta = [{k: torch.tensor(deepcopy(np.array(i[k]))).float() for k in i} for _, i in batch]
    for i in meta:
        i['labels'] = i['labels'].long()
        i['keypoints'] = i['keypoints'].unsqueeze(0)
    return img, meta


def list_img_rec_collate_fn(batch):
    d = {
        'x': [i['x'].unsqueeze(0) for i in batch],
        'label': torch.tensor([i['label'] for i in batch]),
        'index': torch.tensor([i['index'] for i in batch])
    }
    return d

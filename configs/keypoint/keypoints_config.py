import pickle
from pathlib import Path

import imgaug
import numpy as np
import pytorch_lightning
import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets.vision import StandardTransform

from data_loading import CatLMDDataset, CatLMDSubset, SimpleDataset
from models.detection import mobile_net_v3_large_keypoint_rcnn
from utils.collate_fn import key_points_collate_list_fn

seed = 123
pytorch_lightning.seed_everything(seed)
imgaug.seed(seed)

n_epochs = 25
train_batch_size = 16
test_batch_size = 8

train_augmentation = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(),
    # torchvision.transforms.Lambda(aug_combo),
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ColorJitter(),
    torchvision.transforms.RandomAdjustSharpness(0),
    torchvision.transforms.RandomAutocontrast(),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.RandomErasing(),
])
train_augmentation = StandardTransform(train_augmentation, None)

val_augmentation = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
])
val_augmentation = StandardTransform(val_augmentation, None)

dataset = CatLMDDataset(Path('../pets_datasets/CAT_DATASET'))

animals = list(range(len(dataset)))
rand = np.random.RandomState(123)
permuted = rand.permutation(animals)
train_indices = permuted[:int(0.8 * len(permuted))]
val_indices = permuted[int(0.8 * len(permuted)):]

train = CatLMDSubset(dataset, train_indices, train_augmentation, rotate90=True)
val = CatLMDSubset(dataset, val_indices, val_augmentation)
with open('others.pickle', 'rb') as f:
    others = pickle.load(f)
with open('paths.pickle', 'rb') as f:
    paths = pickle.load(f)
dogs_additional_dataset = SimpleDataset('../pets_datasets/data_25', paths, others, train_augmentation, rotate90=True)

with open('others2.pickle', 'rb') as f:
    others = pickle.load(f)
with open('paths2.pickle', 'rb') as f:
    paths = pickle.load(f)
dogs_additional_dataset2 = SimpleDataset('../pets_datasets/data_25', paths, others, train_augmentation, rotate90=True)
train = ConcatDataset((train, dogs_additional_dataset, dogs_additional_dataset2))

p1 = None
p2 = None


def model():
    global p1, p2
    kwargs = {
        "min_size": (320, 336, 352, 368, 384, 400),
        "max_size": 640,
        # "rpn_pre_nms_top_n_test": 20,
        # "rpn_post_nms_top_n_test": 20,
        # "rpn_score_thresh": 0.05,
    }

    model_ = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        pretrained_backbone=True,
        num_classes=2,
        num_keypoints=3,
        box_detections_per_img=1,
        **kwargs
    )
    # model_ = mobile_net_v3_large_keypoint_rcnn()
    p1 = [p for p in model_.backbone.body.parameters() if p.requires_grad]
    p2 = [p for p in model_.parameters() if all(p is not o for o in p1) and p.requires_grad]
    # model_.backbone.requires_grad_(False)
    return model_


class loss(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.coefs = {
            'loss_classifier': 1,
            'loss_box_reg': 1,
            'loss_objectness': 1,
            'loss_rpn_box_reg': 1
        }

    def forward(self, images, targets=None):
        if targets is None:
            return self.model(images)
        x = self.model(images, targets)
        return sum(x.values())


def optimizer(model_: torch.nn.Module):
    optim = torch.optim.AdamW([
        {'params': p1, 'lr': 1e-5},
        {'params': p2}
    ], 1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 15, 20], gamma=0.1)
    return [optim], [sched]


collate_fn = key_points_collate_list_fn


def train_dataloader():
    from torch.utils.data import WeightedRandomSampler
    weights = [1] * (len(train) - len(dogs_additional_dataset) - len(dogs_additional_dataset2)) + \
              [20] * (len(dogs_additional_dataset) + len(dogs_additional_dataset2))
    sampler = WeightedRandomSampler(weights, len(train) + 19 * (len(dogs_additional_dataset) + len(dogs_additional_dataset2)))
    return DataLoader(train, train_batch_size, sampler=sampler, drop_last=True, num_workers=5, collate_fn=collate_fn)


def val_dataloader():
    return [
        DataLoader(
            CatLMDSubset(dataset, train_indices, val_augmentation),
            test_batch_size,
            collate_fn=collate_fn,
            num_workers=4
        ),
        DataLoader(val, test_batch_size, collate_fn=collate_fn, num_workers=4)
    ]


trainer_kwargs = dict(
    move_metrics_to_cpu=True
)

output = Path('results')
output.mkdir(exist_ok=True)

mlflow_target_uri = Path('mlruns')
mlflow_target_uri.mkdir(exist_ok=True)
mlflow_target_uri = str(mlflow_target_uri)
experiment_name = 'LMD'
run_name = 'MobileNetV3 large + eyes + nose + aug_combo 2 + rotate90 + updated bbox + (mini dogs v1+2 x20) '

# devices
device = 'cuda:1'
distributed_train = not isinstance(device, str)
world_size = len(device) if distributed_train else None

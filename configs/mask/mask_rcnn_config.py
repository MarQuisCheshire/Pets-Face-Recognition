from pathlib import Path

import imgaug
import numpy as np
import pytorch_lightning
import torch
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets.vision import StandardTransform

from data_loading import OxfordIIITPet, OxfordSubset
from utils.collate_fn import detection_collate_list_fn

seed = 123
pytorch_lightning.seed_everything(seed)
imgaug.seed(seed)

n_epochs = 100
train_batch_size = 8
test_batch_size = 16

train_augmentation = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(),
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

dataset = OxfordIIITPet(
    root=str(Path('../pets_datasets').resolve()),
    target_types=['body_bbox', 'segmentation'],
)

weights = [(len(dataset.big_classes) - sum(dataset.big_classes)) if i == 1 else sum(dataset.big_classes) for i in
           dataset.big_classes]
animals = list(range(len(dataset)))
rand = np.random.RandomState(123)
val_indices = rand.choice(animals, int(len(animals) * 0.2), replace=False, p=np.array(weights) / np.sum(weights))
train_indices = [i for i in animals if i not in val_indices]
weights = [dataset.big_classes[i] for i in train_indices]
weights = [(len(weights) - sum(weights)) if i == 1 else sum(weights) for i in weights]
sampler = WeightedRandomSampler(weights, 2000)
train = OxfordSubset(dataset, train_indices, train_augmentation, rotate90=True)
val = OxfordSubset(dataset, val_indices, val_augmentation, )

p1 = None
p2 = None


def model():
    global p1, p2
    kwargs = {
        "min_size": 320,
        "max_size": 640,
        # "rpn_pre_nms_top_n_test": 20,
        #     "rpn_post_nms_top_n_test": 20,
        #     "rpn_score_thresh": 0.05,
        'box_detections_per_img': 3
    }

    model_ = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained_backbone=True,
        num_classes=2,
        **kwargs
    )
    model_backbone_p = list(model_.backbone.parameters())
    p1 = [p for p in model_.backbone.parameters() if p.requires_grad]
    p2 = [p for p in model_.parameters() if all(p is not o for o in model_backbone_p) and p.requires_grad]
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


# def loss(config, model):
#     return DETRWrapper(config, model)


def optimizer(model_):
    optim = torch.optim.Adam([
        {'params': p1, 'lr': 1e-6},
        {'params': p2}
    ], 1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[40, 80, 95], gamma=0.1)
    return [optim], [sched]


collate_fn = detection_collate_list_fn


def train_dataloader():
    return DataLoader(train, train_batch_size, sampler=sampler, drop_last=True, num_workers=4, collate_fn=collate_fn)


def val_dataloader():
    return [
        DataLoader(OxfordSubset(dataset, train_indices, val_augmentation, ), test_batch_size,
                   collate_fn=collate_fn,
                   num_workers=4),
        DataLoader(val, test_batch_size, collate_fn=collate_fn, num_workers=4)
    ]


def test_dataloader():
    return [
        DataLoader(val, test_batch_size, collate_fn=collate_fn, num_workers=4),
        DataLoader(OxfordSubset(dataset, train_indices, val_augmentation, ), test_batch_size,
                   collate_fn=collate_fn,
                   num_workers=4)
    ]


trainer_kwargs = dict(
    move_metrics_to_cpu=True
)

output = Path('results')
output.mkdir(exist_ok=True)

mlflow_target_uri = Path('mlruns')
mlflow_target_uri.mkdir(exist_ok=True)
mlflow_target_uri = str(mlflow_target_uri)
experiment_name = 'Body Detection'
run_name = 'MASK R-CNN + rotate90 + 3 NMS'

# devices
device = 'cuda:0'
distributed_train = not isinstance(device, str)
world_size = len(device) if distributed_train else None

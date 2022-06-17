from pathlib import Path

import numpy as np
import pytorch_lightning
import torch
import torch.nn.functional as F
import torchvision.models.mobilenetv2
from torch.utils.data import DataLoader, ConcatDataset

from data_loading import RecDataset, PairGenerator, RecSubset
from data_loading.dataset import simple_init_dataset
from losses import SoftmaxBasedMetricLearning
from utils.preprocs import resize_with_padding

seed = 123
pytorch_lightning.seed_everything(seed)

train_augmentation = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Lambda(resize_with_padding),
    torchvision.transforms.RandomCrop((252, 252)),
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.RandomRotation(5),
    torchvision.transforms.RandomAdjustSharpness(0, 0.1),
    torchvision.transforms.RandomAutocontrast(0.3),
    torchvision.transforms.ToTensor(),
])

val_augmentation = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Lambda(resize_with_padding),
    torchvision.transforms.ToTensor(),
])

dataset = RecDataset(
    Path('../pets_datasets/data_25_transformed_v4_masked_cats'),
    None,
    2,
    init_dataset_method=simple_init_dataset
)

perm = np.random.RandomState(seed).permutation(dataset.get_users())
tr_size = (len(perm) - 3100) / len(perm)
train_users = [perm[i] for i in range(int(len(perm) * tr_size))]
val_users = [perm[i] for i in range(int(len(perm) * tr_size), len(perm))]
train_indices = [j for i in train_users for j in dataset.uid_to_indices[i]]
val_indices = [j for i in val_users for j in dataset.uid_to_indices[i]]

assert len(set(train_indices) & set(val_indices)) == 0

dataset3 = RecDataset(
    Path('../pets_datasets/petfinder_extra_cats_transformed_v4_masked'),
    None,
    3,
    init_dataset_method=simple_init_dataset,
    start_class=len(train_users)
)
dataset3 = RecSubset(dataset3, list(range(len(dataset3))), train_augmentation)

train = RecSubset(dataset, train_indices, train_augmentation)
train = ConcatDataset((train, dataset3))
val = RecSubset(dataset, val_indices, val_augmentation)
for a, b in enumerate(train_users):
    dataset.label_map[b] = a

__pair_gen = PairGenerator(dataset, 10000, 1, None, seed, val_users)

n_epochs = 50
train_batch_size = 50
test_batch_size = 20

thrs = np.linspace(0.5, 0.99, 6)
far_thr = [0.1, 0.05, 0.03, 0.01, 0.005, 0.001]
k = [5, 10, 100]

print(len(val_users), len(val_users) + len(train_users))
print(len(val_indices), len(val_indices) + len(train_indices))
print(np.sum(__pair_gen.labels), len(__pair_gen))


def pair_generator(idx):
    if idx == 0:
        return 'Val', __pair_gen
    raise Exception


def similarity_f(pairs):
    t1 = torch.cat([i[0].unsqueeze(0) for i in pairs], dim=0)
    t2 = torch.cat([i[1].unsqueeze(0) for i in pairs], dim=0)

    return (F.cosine_similarity(t1, t2) + 1) / 2


def model():
    # model_ = torchvision.models.mobilenet_v2(pretrained=True)
    # model_.classifier = torch.nn.Sequential(
    #     torch.nn.Linear(model_.last_channel, 512),
    # )
    #
    model_ = torchvision.models.resnet50(pretrained=True)
    model_.fc = torch.nn.Linear(2048, 512)

    # model_ = torchvision.models.efficientnet_b2(pretrained=True)
    # model_.classifier = torch.nn.Linear(1408, 512)
    # model_ = torchvision.models.convnext_tiny(pretrained=True)
    # model_.classifier[2] = torch.nn.Linear(768, 512)
    return model_


def loss(config, model_):
    _ = config
    return SoftmaxBasedMetricLearning(
        model=model_,
        num_class=len(dataset3.dataset.get_users()) + len(train_users),
        embedding_size=512,
        is_focal=True,
        arc_margin=True
    )


def optimizer(model_):
    params1 = [p for i, p in model_.module.named_parameters() if 'fc' not in i]
    params2 = [p for i, p in model_.module.named_parameters() if 'fc' in i]
    d = [
        {'lr': 10 ** -4 / 2, 'params': params1},
        {'lr': 10 ** -4, 'params': params2},
        {'lr': 10 ** -4, 'params': model_.add_margin.parameters(), 'weight_decay': 1 * (10 ** -4)}
    ]
    optim = torch.optim.AdamW(d, 1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[35, 45], gamma=0.1)
    return [optim], [sched]


def train_dataloader():
    return DataLoader(train, train_batch_size, shuffle=True, drop_last=True, num_workers=5)


def val_dataloader():
    return DataLoader(val, test_batch_size, num_workers=0)


trainer_kwargs = dict(
    benchmark=True,
    # gradient_clip_val=1.,
    # gradient_clip_algorithm='norm',
)

output = Path('results')
output.mkdir(exist_ok=True)

mlflow_target_uri = Path('mlruns')
mlflow_target_uri.mkdir(exist_ok=True)
mlflow_target_uri = str(mlflow_target_uri)
experiment_name = 'Cats (Body)'
run_name = 'Updated ResNet50 datasetv4 MASKED without jitter'

# devices
device = 'cuda:1'
distributed_train = not isinstance(device, str)
world_size = len(device) if distributed_train else None

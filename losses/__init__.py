import torch
import torch.nn as nn

from .large_margin import ArcMarginProduct, AddMarginProduct
from .losses import FocalLoss


class SoftmaxBasedMetricLearning(nn.Module):

    def __init__(
            self,
            model: nn.Module,
            num_class,
            embedding_size=512,
            s=64.0,
            m=0.5,
            is_focal=False,
            loss_kwargs=None,
            arc_margin=False,
            easy_margin=False,
    ):
        super().__init__()
        if arc_margin:
            self.add_margin = ArcMarginProduct(embedding_size, num_class, s=s, m=m, easy_margin=easy_margin)
        else:
            self.add_margin = AddMarginProduct(embedding_size, num_class, s=s, m=m)

        if loss_kwargs is None:
            loss_kwargs = {}
        if is_focal:
            self.focal_loss = FocalLoss(num_class=num_class, **loss_kwargs)
        else:
            self.focal_loss = nn.CrossEntropyLoss(**loss_kwargs)
        self.module = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img, label=None, **__):
        if isinstance(img, (list, tuple)):
            tensor = torch.cat([self.module(i) for i in img], dim=0)
        else:
            tensor = self.module(img)
        if label is None:
            return tensor
        logits = self.add_margin(tensor, label)
        loss = self.focal_loss(logits, label)
        return {'loss': loss, 'emb': tensor, 'logits': logits}


class DummyWrapper(nn.Module):
    def __init__(self, model, *_, **__):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

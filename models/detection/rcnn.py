import torch.nn
import torchvision
from torchvision.models.detection import FasterRCNN, KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import FrozenBatchNorm2d
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from ..swin import swin_t


def mobile_net_v3_large_rcnn():
    kwargs = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }
    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
    backbone = torchvision.models.mobilenet_v3_large(pretrained=True, norm_layer=FrozenBatchNorm2d).features
    stage_indices = list(range(len(backbone)))
    num_stages = len(stage_indices)
    extra_blocks = LastLevelMaxPool()
    returned_layers = [num_stages - 2, num_stages - 1]
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
    out_channels = 256
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model_ = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        box_detections_per_img=1,
        **kwargs
    )

    return model_


def convnetx_tiny_rcnn():
    kwargs = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }
    backbone = torchvision.models.convnext_tiny(pretrained=True, stochastic_depth_prob=0).features
    stage_indices = list(range(len(backbone)))
    num_stages = len(stage_indices)
    extra_blocks = LastLevelMaxPool()
    returned_layers = [num_stages - 3, num_stages - 1]
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [384, 768]
    out_channels = 256
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
    aspect_ratios = ((10 / 14, 1.0, 14 / 10),) * len(anchor_sizes)

    model_ = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        box_detections_per_img=1,
        **kwargs
    )

    return model_


def convnext_tiny_keypoint_rcnn():
    kwargs = {
        "min_size": (320, 336, 352, 368, 384, 400),
        "max_size": 640,
        "box_detections_per_img": 1
    }

    backbone = torchvision.models.convnext_tiny(pretrained=True, stochastic_depth_prob=0).features

    stage_indices = list(range(len(backbone)))
    num_stages = len(stage_indices)
    extra_blocks = LastLevelMaxPool()
    returned_layers = [num_stages - 7, num_stages - 5, num_stages - 3, num_stages - 1]
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [96, 192, 384, 768]
    out_channels = 256
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    model_ = KeypointRCNN(
        backbone,
        num_classes=2,
        num_keypoints=3,
        **kwargs
    )
    return model_


def mobile_net_v3_large_keypoint_rcnn():
    kwargs = {
        "min_size": (320, 336, 352, 368, 384, 400),
        "max_size": 640,
        "box_detections_per_img": 1
    }
    backbone = torchvision.models.mobilenet_v3_large(pretrained=True, norm_layer=FrozenBatchNorm2d).features

    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)
    extra_blocks = LastLevelMaxPool()
    returned_layers = [num_stages - 4, num_stages - 3, num_stages - 2, num_stages - 1]
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
    out_channels = 256
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    model_ = KeypointRCNN(
        backbone,
        num_classes=2,
        num_keypoints=3,
        **kwargs
    )
    return model_


def swin_tiny_keypoint_rcnn():
    kwargs = {
        "min_size": (320, 336, 352, 368, 384, 400),
        "max_size": 640,
        "box_detections_per_img": 1
    }
    backbone = swin_t()
    backbone = torch.nn.Sequential(
        backbone.stage1,
        backbone.stage2,
        backbone.stage3,
        backbone.stage4
    )

    stage_indices = list(range(len(backbone)))
    num_stages = len(stage_indices)
    extra_blocks = LastLevelMaxPool()
    returned_layers = [num_stages - 4, num_stages - 3, num_stages - 2, num_stages - 1]
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [96, 96 * 2, 96 * 4, 96 * 8]
    out_channels = 256
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    model_ = KeypointRCNN(
        backbone,
        num_classes=2,
        num_keypoints=3,
        **kwargs
    )
    model_.transform = GeneralizedRCNNTransform(
        343,
        686,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        size_divisible=7 * 7
    )
    return model_

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead

from ..builder import HEADS
import torch
from torch import nn, Tensor
from typing import Optional, Tuple
from ..parcnet.layers import ConvLayer, SeparableConv
from ..parcnet.misc.profiler import module_profile
from ..parcnet.misc.init_utils import initialize_conv_layer


@HEADS.register_module()
class ParCNetHead(BBoxHead):
    """
        This class defines the SSD object detection Head
            https://arxiv.org/abs/1512.02325
    """
    def __init__(self,
                 in_channels: int,
                 n_anchors: int,
                 n_classes: int,
                 n_coordinates: int,
                 model_activation_name: str = "prelu",
                 model_activation_neg_slope: float = 0.1,
                 model_activation_inplace: bool = False,
                 model_normalization_name: str = "batch_norm",
                 model_normalization_groups: int = 1,
                 model_normalization_momentum: float = 0.1,
                 proj_channels: Optional[int] = -1,
                 kernel_size: Optional[int] = 3, *args, **kwargs) -> None:
        super().__init__()
        self.in_channel = in_channels

        proj_layer = None
        self.proj_channels = None
        if proj_channels != -1 and proj_channels != in_channels and kernel_size and kernel_size > 1:
            proj_layer = ConvLayer(
                in_channels=in_channels, out_channels=proj_channels or -1, kernel_size=1,
                stride=1, groups=1, bias=False, use_norm=True, use_act=True,
                model_activation_name=model_activation_name,
                model_activation_neg_slope=model_activation_neg_slope,
                model_activation_inplace=model_activation_inplace,
                model_normalization_name=model_normalization_name,
                model_normalization_groups=model_normalization_groups,
                model_normalization_momentum=model_normalization_momentum
            )
            in_channels = proj_channels or -1
            self.proj_channels = proj_channels

        self.proj_layer = proj_layer

        conv_fn = ConvLayer if kernel_size == 1 else SeparableConv
        self.loc_cls_layer = conv_fn(in_channels=in_channels,
                                     out_channels=n_anchors * (n_coordinates + n_classes),
                                     kernel_size=kernel_size or 3, stride=1, groups=1, bias=True, use_norm=False,
                                     use_act=False,
                                     model_activation_name=model_activation_name,
                                     model_activation_neg_slope=model_activation_neg_slope,
                                     model_activation_inplace=model_activation_inplace,
                                     model_normalization_name=model_normalization_name,
                                     model_normalization_groups=model_normalization_groups,
                                     model_normalization_momentum=model_normalization_momentum)

        self.n_coordinates = n_coordinates
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.k_size = kernel_size

        self.reset_parameters()

    def __repr__(self):
        repr_str = "{}(in_channels={}, n_anchors={}, n_classes={}, n_coordinates={}, kernel_size={}".format(
            self.__class__.__name__,
            self.in_channel,
            self.n_anchors,
            self.n_classes,
            self.n_coordinates,
            self.k_size
        )
        if self.proj_layer is not None:
            repr_str += ", proj=True, proj_channels={}".format(self.proj_channels)

        repr_str += ")"
        return repr_str

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method='xavier_uniform')

    def forward(self, x: Tensor):
        batch_size = x.shape[0]

        if self.proj_layer is not None:
            x = self.proj_layer(x)

        # [B x C x H x W] --> [B x Anchors * (coordinates + classes) x H x W]
        x = self.loc_cls_layer(x)

        # [B x Anchors * (coordinates + classes) x H x W] --> [B x H x W x Anchors * (coordinates + classes)]
        x = x.permute(0, 2, 3, 1)
        # [B x H x W x Anchors * (coordinates + classes)] --> [B x H*W*Anchors X (coordinates + classes)]
        x = x.contiguous().view(batch_size, -1, self.n_coordinates + self.n_classes)

        # [B x H*W*Anchors X (coordinates + classes)] --> [B x H*W*Anchors X coordinates], [B x H*W*Anchors X classes]
        box_locations, box_classes = torch.split(x, [self.n_coordinates, self.n_classes], dim=-1)
        return box_classes, box_locations

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        params = macs = 0.0

        if self.proj_layer is not None:
            input, p, m = module_profile(module=self.proj_layer, x=input)
            params += p
            macs += m

        x, p, m = module_profile(module=self.loc_cls_layer, x=input)
        params += p
        macs += m

        return input, params, macs

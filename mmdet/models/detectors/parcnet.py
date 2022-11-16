# Copyright (c) OpenMMLab. All rights reserved.
import logging
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS
# from ..builder import build_backbone, build_head, build_neck
from .base import BaseDetector

from torch import nn, Tensor
from typing import Optional, Tuple, Dict, Union
from torchvision.ops import nms as torch_nms

from ..parcnet.models.detection.base_detection import DetectionPredTuple
from ..parcnet.layers import ConvLayer, SeparableConv, AdaptiveAvgPool2d
from ..parcnet.modules import SSDHead
from ..parcnet.models.classification import build_classification_model
from ..parcnet.misc.init_utils import initialize_conv_layer
from ..parcnet.misc import box_utils
from ..parcnet.misc.anchor_generator import SSDAnchorGenerator
from ..parcnet.misc.profiler import module_profile
from ..parcnet.misc.common import parameter_list
from ..parcnet.misc.init_utils import initialize_weights


@DETECTORS.register_module()
class ParCNet(BaseDetector):
    """
    This class implements Single Shot Object Detector
        https://arxiv.org/abs/1512.02325
    """
    coordinates = 4 # 4 coordinates (x, y, x1, y1) or (x, y, w, h)

    def __init__(self,
                 opts,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 backbone=None,
                 bbox_head=None,
                 neck=None):
        
        logging.warning('---------------------------')
        super().__init__(init_cfg)
        logging.warning('---------------------------')
        self.n_detection_classes = getattr(opts, 'model_detection_n_classes', 80)
        
        output_stride = opts.get("model_detection_output_stride", None)
        self.encoder = build_classification_model(
            opts=opts,
            output_stride=output_stride
        )

        enc_conf = self.encoder.model_conf_dict

        logging.warning('---------------------------')
        enc_ch_l5_out_proj = _check_out_channels(enc_conf, "exp_before_cls")
        enc_ch_l5_out = _check_out_channels(enc_conf, 'layer5')
        enc_ch_l4_out = _check_out_channels(enc_conf, 'layer4')
        enc_ch_l3_out = _check_out_channels(enc_conf, 'layer3')
        enc_ch_l2_out = _check_out_channels(enc_conf, 'layer2')
        enc_ch_l1_out = _check_out_channels(enc_conf, 'layer1')

        logging.warning('---------------------------')
        self.enc_l5_channels = enc_ch_l5_out
        self.enc_l5_channels_exp = enc_ch_l5_out_proj
        self.enc_l4_channels = enc_ch_l4_out
        self.enc_l3_channels = enc_ch_l3_out
        self.enc_l2_channels = enc_ch_l2_out
        self.enc_l1_channels = enc_ch_l1_out

        logging.warning('---------------------------')
        self.n_classes = getattr(opts, 'model_detection_n_classes', 80)
        self.opts = opts
        
        # =======================

        # delete layers that are not required in detection network
        self.encoder.classifier = None
        self.encoder.conv_1x1_exp = None
        logging.warning('---------------------------')

        output_strides = opts.get("model_detection_ssd_output_strides", [16, 32, 64, 128, 256, -1 ])
        n_os = len(output_strides)

        anchors_aspect_ratio = opts.get("model_detection_ssd_anchors_aspect_ratio", [[2, 3]] * len(output_strides))
        proj_channels = opts.get("model_detection_ssd_proj_channels", [512, 256, 256, 128, 128, 64])

        anchors_aspect_ratio = anchors_aspect_ratio + [[2]] * (n_os - len(anchors_aspect_ratio))
        proj_channels = proj_channels + [128] * (n_os - len(proj_channels))

        logging.warning('---------------------------')
        if len(output_strides) != len(anchors_aspect_ratio) != len(proj_channels):
            warnings.warn(
                "SSD model requires anchors to be defined for feature maps from each output stride. Also"
                "len(anchors_aspect_ratio) == len(output_strides) == len(proj_channels). "
                "Got len(output_strides)={}, len(anchors_aspect_ratio)={}, len(proj_channels)={}."
                " Please specify correct arguments using following arguments: "
                "\n--model.detection.ssd.anchors-aspect-ratio "
                "\n--model.detection.ssd.output-strides"
                "\n--model.detection.ssd.proj-channels".format(
                    len(output_strides),
                    len(anchors_aspect_ratio),
                    len(proj_channels)
                )
            )
        logging.warning('---------------------------')
        extra_layers = {}
        enc_channels_list = []
        in_channels = self.enc_l5_channels

        extra_proj_list = [256] * (len(output_strides) - len(proj_channels))
        proj_channels = proj_channels + extra_proj_list
        for idx, os in enumerate(output_strides):
            out_channels = proj_channels[idx]
            if os == 8:
                enc_channels_list.append(self.enc_l3_channels)
            elif os == 16:
                enc_channels_list.append(self.enc_l4_channels)
            elif os == 32:
                enc_channels_list.append(self.enc_l5_channels)
            elif os > 32 and os != -1:
                extra_layers["os_{}".format(os)] = SeparableConv(
                    opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=3, use_act=True,
                    use_norm=True, stride=2
                )
                enc_channels_list.append(out_channels)
                in_channels = out_channels
            elif os == -1:
                extra_layers["os_{}".format(os)] = nn.Sequential(
                    AdaptiveAvgPool2d(output_size=1),
                    ConvLayer(opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              use_act=True, use_norm=False)
                )
                enc_channels_list.append(out_channels)
                in_channels = out_channels
            else:
                raise NotImplementedError
        self.extra_layers = None if not extra_layers else nn.ModuleDict(extra_layers)
        if self.extra_layers is not None:
            self.reset_layers(module=self.extra_layers)

        logging.warning('---------------------------')
        # Anchor box related parameters
        self.conf_threshold = opts.get("model_detection_ssd_conf_threshold", 0.01)
        self.nms_threshold = opts.get("model_detection_ssd_nms_iou_threshold", 0.3)
        self.top_k = opts.get("model_detection_ssd_num_objects_per_class", 200)

        self.anchor_box_generator = SSDAnchorGenerator(
            output_strides=output_strides,
            aspect_ratios=anchors_aspect_ratio,
            min_ratio=opts.get("model_detection_ssd_min_box_size", 0.1),
            max_ratio=opts.get("model_detection_ssd_max_box_size", 1.05)
        )

        logging.warning('---------------------------')
        anchors_aspect_ratio = self.anchor_box_generator.num_anchors_per_os()
        self.ssd_heads = nn.ModuleList()

        logging.warning('---------------------------')
        # Create SSD detection and classification heads
        for os, in_dim, proj_dim, n_anchors in zip(output_strides, enc_channels_list, proj_channels, anchors_aspect_ratio):
            self.ssd_heads += [
                SSDHead(opts=opts,
                        in_channels=in_dim,
                        n_classes=self.n_classes,
                        n_coordinates=self.coordinates,
                        n_anchors=n_anchors,
                        proj_channels=proj_dim,
                        kernel_size=3 if os != -1 else 1)
            ]

        self.anchors_aspect_ratio = anchors_aspect_ratio
        self.output_strides = output_strides
        
        logging.warning('---------------------------')

    @staticmethod
    def reset_layers(module):
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method='xavier_uniform')

    @staticmethod
    def process_anchors_ar(anchor_ar):
        assert isinstance(anchor_ar, list)
        new_ar = []
        for ar in anchor_ar:
            if ar in new_ar:
                continue
            new_ar.append(ar)
        return new_ar

    def ssd_forward(self, x: Tensor, *args, **kwargs) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        # print("****************************************************************************************ssd_forward ****")
        enc_end_points: Dict = self.encoder.extract_end_points_all(x)

        is_prediction = kwargs.get("is_prediction", False)

        locations = []
        confidences = []
        anchors = None if not is_prediction else []

        x = enc_end_points["out_l5"]
        for os, ssd_head in zip(self.output_strides, self.ssd_heads):
            if os == 8:
                fm_h, fm_w = enc_end_points["out_l3"].shape[2:]
                loc, pred = ssd_head(enc_end_points["out_l3"])
            elif os == 16:
                fm_h, fm_w = enc_end_points["out_l4"].shape[2:]
                loc, pred = ssd_head(enc_end_points["out_l4"])
            elif os == 32:
                fm_h, fm_w = enc_end_points["out_l5"].shape[2:]
                loc, pred = ssd_head(enc_end_points["out_l5"])
            else: # for all other feature maps with os > 32
                x = self.extra_layers["os_{}".format(os)](x)
                fm_h, fm_w = x.shape[2:]
                loc, pred = ssd_head(x)
            locations.append(loc)
            confidences.append(pred)

            if anchors is not None:
                # anchors in center form
                anchors_fm_ctr = self.anchor_box_generator(
                    fm_height=fm_h,
                    fm_width=fm_w,
                    fm_output_stride=os
                )
                anchors.append(anchors_fm_ctr)

        locations = torch.cat(locations, dim=1)
        confidences = torch.cat(confidences, dim=1)

        if anchors is None:
            # return a tuple because dictionary is not supported on CoreML
            return confidences, locations

        anchors = torch.cat(anchors, dim=0)
        anchors = anchors.unsqueeze(dim=0).to(device=x.device)
        return confidences, locations, anchors

    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        return self.ssd_forward(x=x)

    def predict(self, x: Tensor, *args, **kwargs):
        bsz, channels, width, height = x.shape
        assert bsz == 1

        with torch.no_grad():
            confidences, locations, anchors = self.ssd_forward(x, is_prediction=True)
            scores = nn.Softmax(dim=-1)(confidences)
            # convert boxes in center form [c_x, c_y]
            boxes = box_utils.convert_locations_to_boxes(
                pred_locations=locations,
                anchor_boxes=anchors,
                center_variance=getattr(self.opts, "model.detection.ssd.center_variance", 0.1),
                size_variance=getattr(self.opts, "model.detection.ssd.size_variance", 0.2)
            )
            # convert boxes from center form [c_x, c_y] to corner form [x, y]
            boxes = box_utils.center_form_to_corner_form(boxes)

        # post-process the boxes and scores
        boxes = boxes[0] # remove the batch dimension
        scores = scores[0]

        object_labels = []
        object_boxes = []
        object_scores = []
        for class_index in range(1, self.n_classes):
            probs = scores[:, class_index]
            mask = probs > self.conf_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            masked_boxes = boxes[mask, :]

            filtered_boxes, filtered_scores = nms(
                scores=probs.reshape(-1),
                boxes=masked_boxes,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k
            )
            object_boxes.append(filtered_boxes)
            object_scores.append(filtered_scores)
            object_labels.extend([class_index] * filtered_boxes.size(0))

        # no object detected
        if not object_scores:
            return DetectionPredTuple(
                labels=torch.empty(0),
                scores=torch.empty(0),
                boxes=torch.empty(0, 4)
            )

        # concatenate all results
        object_scores = torch.cat(object_scores)
        object_boxes = torch.cat(object_boxes)
        object_labels = torch.tensor(object_labels)

        return DetectionPredTuple(
            labels=object_labels,
            scores=object_scores,
            boxes=object_boxes
        )

    def profile_model(self, input: Tensor) -> None:
        # Note: Model profiling is for reference only and may contain errors.
        # It relies heavily on the user to implement the underlying functions accurately.

        overall_params, overall_macs = 0.0, 0.0

        warnings.warn('Model statistics for an input of size {}'.format(input.size()))
        warnings.warn('-'*65)
        print('{:>35} Summary'.format(self.__class__.__name__))
        warnings.warn('-'*65)

        # profile encoder
        # enc_str = logger.text_colors['logs'] + logger.text_colors['bold'] + 'Encoder  ' + logger.text_colors[
        #     'end_color']
        # print('{:>45}'.format(enc_str))
        enc_end_points, encoder_params, encoder_macs = self.encoder.profile_model(input, is_classification=False)

        ssd_head_params = ssd_head_macs = 0.0
        x = enc_end_points["out_l5"]
        for os, ssd_head in zip(self.output_strides, self.ssd_heads):
            if os == 8:
                _, p, m = module_profile(module=ssd_head, x=enc_end_points["out_l3"])
                ssd_head_params += p
                ssd_head_macs += m
            elif os == 16:
                _, p, m = module_profile(module=ssd_head, x=enc_end_points["out_l4"])
                ssd_head_params += p
                ssd_head_macs += m
            elif os == 32:
                _, p, m = module_profile(module=ssd_head, x=enc_end_points["out_l5"])
                ssd_head_params += p
                ssd_head_macs += m
            else: # for all other feature maps with os > 32
                x, p1, m1 = module_profile(module=self.extra_layers["os_{}".format(os)], x=x)
                _, p2, m2 = module_profile(module=ssd_head, x=x)
                ssd_head_params += (p1 + p2)
                ssd_head_macs += (m1 + m2)

        overall_params += (encoder_params + ssd_head_params)
        overall_macs += (encoder_macs + ssd_head_macs)

        # ssd_str = logger.text_colors['logs'] + logger.text_colors['bold'] + 'SSD  ' + logger.text_colors[
        #     'end_color']
        # print('{:>45}'.format(ssd_str))

        print(
            '{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M'.format(
                self.__class__.__name__,
                'Params',
                round(ssd_head_params / 1e6, 3),
                'MACs',
                round(ssd_head_macs / 1e6, 3)
            )
        )

        warnings.warn('-'*65)
        print('{:<20} = {:>8.3f} M'.format('Overall parameters', overall_params / 1e6))
        # Counting Addition and Multiplication as 1 operation
        print('{:<20} = {:>8.3f} M'.format('Overall MACs', overall_macs / 1e6))
        overall_params_py = sum([p.numel() for p in self.parameters()])
        print('{:<20} = {:>8.3f} M'.format('Overall parameters (sanity check)', overall_params_py / 1e6))
        warnings.warn('-'*65)


# --------------------------- SSD to override

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super().forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    

    # @classmethod
    # def add_arguments(cls, parser: argparse.ArgumentParser):
    #     return parser

    @staticmethod
    def reset_layer_parameters(layer, opts):
        # weight initialization
        initialize_weights(opts=opts, modules=layer.modules())

    def get_trainable_parameters(self, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False):
        param_list = parameter_list(named_parameters=self.named_parameters,
                                    weight_decay=weight_decay,
                                    no_decay_bn_filter_bias=no_decay_bn_filter_bias)
        return param_list, [1.0] * len(param_list)

    @staticmethod
    def profile_layer(layer, input):
        block_params = block_macs = 0.0
        if isinstance(layer, nn.Sequential):
            for layer_i in range(len(layer)):
                input, layer_param, layer_macs = layer[layer_i].profile_module(input)
                block_params += layer_param
                block_macs += layer_macs
                print(
                    '{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M'.format(layer[layer_i].__class__.__name__,
                                                                              'Params',
                                                                              round(layer_param / 1e6, 3),
                                                                              'MACs',
                                                                              round(layer_macs / 1e6, 3)
                                                                              )
                )
        else:
            input, layer_param, layer_macs = layer.profile_module(input)
            block_params += layer_param
            block_macs += layer_macs
            print('{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M'.format(layer.__class__.__name__,
                                                                            'Params',
                                                                            round(layer_param / 1e6, 3),
                                                                            'MACs',
                                                                            round(layer_macs / 1e6, 3)))
        return input, block_params, block_macs


def nms(boxes: Tensor, scores: Tensor, nms_threshold: float, top_k: Optional[int] = 200) -> Tuple[Tensor, Tensor]:
    """
    Args:
        boxes (N, 4): boxes in corner-form.
        scores (N): probabilities
        nms_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked: Boxes and scores
    """
    keep = torch_nms(boxes, scores, nms_threshold)
    if top_k > 0:
        keep = keep[:top_k]
    return boxes[keep], scores[keep]

def _check_out_channels(config: Dict, layer_name: str) -> int:
    enc_ch_l: Dict = config.get(layer_name, None)
    if enc_ch_l is None or not enc_ch_l:
        warnings.warn('Encoder does not define input-output mapping for {}: Got: {}'.format(layer_name, config))

    enc_ch_l_out = enc_ch_l.get('out', None)
    if enc_ch_l_out is None or not enc_ch_l_out:
        warnings.warn(
            'Output channels are not defined in {} of the encoder. Got: {}'.format(layer_name, enc_ch_l))

    return enc_ch_l_out

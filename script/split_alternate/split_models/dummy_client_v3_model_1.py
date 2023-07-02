'''This file contains the split versions (in ClientModel and ServerModel) of
configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml'''

import warnings
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Tuple
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torchvision.transforms.functional as Ft
from torchdistill.common import tensor_util
from torchvision.models.detection.anchor_utils import AnchorGenerator

# ==================================================== Client ====================================================

def dequantize_tensor(quantized_tensor, scale, zero_point, cuda = True):
    zero_point_tensor = torch.full(quantized_tensor.shape, zero_point.item(), dtype=quantized_tensor.dtype)
    scale_tensor = torch.full(quantized_tensor.shape, scale.item(), dtype=quantized_tensor.dtype)
    if cuda:
        zero_point_tensor = zero_point_tensor.cuda()
        scale_tensor = scale_tensor.cuda()
    return quantized_tensor.sub(zero_point_tensor).mul(scale_tensor)

class ClientModel(nn.Module):
    '''client model for faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml'''

    def __init__(self, encoder):
        super(ClientModel, self).__init__()
        # self.transform = student_model.transform
        self.encoder = encoder
        # self.training = student_model.training
        # self.compressor = student_model.backbone.body.bottleneck_layer.compressor

    def transform_forward(self, image: Tensor):
        input_preprocessed = image
        print(image.shape)
        if image.shape[0] == 3:
            input_preprocessed = Ft.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            input_preprocessed = Ft.normalize(image, mean=[0.5], std=[0.5])
        # input_preprocessed = Ft.resize(input_preprocessed,(800,), Ft.InterpolationMode.BILINEAR, 1333, "warn")
        input_preprocessed = Ft.resize(input_preprocessed,(400,), Ft.InterpolationMode.BILINEAR, 1333, "warn")
        # input_preprocessed = Ft.resize(input_preprocessed,(350,), Ft.InterpolationMode.BILINEAR, 1333, "warn")
        # input_preprocessed = Ft.resize(input_preprocessed,(300,), Ft.InterpolationMode.BILINEAR, 1333, "warn")
        return input_preprocessed
    
    def quantize_tensor(self, x, cuda = True, num_bits=8):
        assert torch.is_tensor(x)
        qmin = 0.
        qmax = 2.**num_bits - 1.
        min_val, max_val = x.min(), x.max()

        scale = (max_val.item() - min_val.item()) / (qmax - qmin)
        initial_zero_point = qmin - min_val / scale

        zero_point = 0
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point.item()


        # zero_point = int(zero_point)
        zero_point_tensor = torch.full(x.shape,zero_point, dtype=x.dtype)
        scale_tensor = torch.full(x.shape,scale, dtype=x.dtype)
        if cuda:
            zero_point_tensor = zero_point_tensor.cuda()
            scale_tensor = scale_tensor.cuda()
        q_x = x.div(scale_tensor).add(zero_point_tensor)
        q_x.clamp_(qmin, qmax).round_()

        q_x = q_x.to(x.dtype)
        scale = torch.tensor(scale, dtype=x.dtype)
        zero_point = torch.tensor(zero_point, dtype=x.dtype)

        return (q_x, scale, zero_point)

    def encoder_forward(self, z):
        z = self.encoder(z)
        z = self.quantize_tensor(z)
        return z
    
    def forward(self, x):
        reshaped_image = self.transform_forward(x)
        # reshaped_image = x
        features = self.encoder_forward(reshaped_image.unsqueeze(0))
        # return features[0].to(torch.float32).detach().cuda()
        quantized_tensor = features[0].to(torch.float32).detach().cuda()
        scale = features[1].to(torch.float32).detach().cuda()
        zero_point = torch.tensor(features[2], dtype=torch.float32).cuda()
        return quantized_tensor, scale, zero_point

# ==================================================== SERVER ====================================================

# RPN Helper functions used for redefining RPN Forward
# RPN Forward currently takes in the images as input (while only using the shapes); by re-defining the function,
# we avoid using the images as input. However, our custom re-definition must have these non-built in functions
# from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/rpn.py
def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def rpn_forward(rpn: AnchorGenerator, features, targets, image_sizes, secondary_image_size):
    features = list(features.values())
    objectness, pred_bbox_deltas = rpn.head(features)

    # modify anchor generator to not use images
    anchors = rpn.anchor_generator(image_sizes, secondary_image_size, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)

    # modified to use image_sizes
    boxes, scores = rpn.filter_proposals(proposals, objectness, image_sizes, num_anchors_per_level)

    losses = {}
    if rpn.training:
        if targets is None:
            raise ValueError("targets should not be None")
        labels, matched_gt_boxes = rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
    return boxes, losses


def anchor_forward(rpn, image_sizes, tensorshapes, feature_maps: List[Tensor]) -> List[Tensor]:
    grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
    image_size = tensorshapes

    dtype, device = feature_maps[0].dtype, feature_maps[0].device
    strides = [
        [
            torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
            torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
        ]
        for g in grid_sizes
    ]

    print("type rpn: ", type(rpn))

    rpn.set_cell_anchors(dtype, device)
    anchors_over_all_feature_maps = rpn.grid_anchors(grid_sizes, strides)
    anchors: List[List[torch.Tensor]] = []
    for _ in range(len(image_sizes)):
        anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
        anchors.append(anchors_in_image)
    anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
    return anchors


class ServerModel(nn.Module):
    def __init__(self, student_model2):
        super().__init__()
        student_model = deepcopy(student_model2)
        self.backbone = student_model.backbone

        # change bottleneck_layer to only decode (encoder in ClientModel)
        self.backbone.body.bottleneck_layer.encoder = nn.Identity()
        self.backbone.body.bottleneck_layer.compressor = nn.Identity()
        self.backbone.body.bottleneck_layer.decompressor = tensor_util.dequantize_tensor
        self.backbone.body.bottleneck_layer.encode = lambda z: {'z': z}

        # change RPN and anchor generator
        self.rpn = student_model.rpn

        self.rpn.anchor_generator.forward = lambda image_sizes, tensorshapes, feature_maps: anchor_forward(
            self.rpn.anchor_generator, image_sizes, tensorshapes, feature_maps)
        self.rpn.forward = lambda features, targets, image_sizes, secondary_image_size: rpn_forward(self.rpn, features,
                                                                                                    targets,
                                                                                                    image_sizes,
                                                                                                    secondary_image_size)

        # def anchor_generator_custom_forward(self, image_sizes, tensorshapes, feature_maps):
        #     ancor : AnchorGenerator = self.rpn.anchor_generator
        #     return anchor_forward(ancor, image_sizes, tensorshapes, feature_maps)
        
        # def rpn_custom_forward(self, features, targets, image_sizes, secondary_image_size):
        #     return rpn_forward(self.rpn, features, targets, image_sizes, secondary_image_size)


        # self.rpn.anchor_generator.forward = anchor_generator_custom_forward
        # self.rpn.forward = rpn_custom_forward

        self.roi_heads = student_model.roi_heads
        self.postprocess = student_model.transform.postprocess
        self._has_warned = student_model._has_warned
        self.eager_outputs = student_model.eager_outputs


    # def forward(self, features, targets, image_sizes, original_image_sizes, secondary_image_size):
    def forward(self, quantized_tensor, scale, zero_point, image_sizes, original_image_sizes, secondary_image_size):
        '''image sizes '''
        # targets is None
        features = self.backbone((quantized_tensor, scale, zero_point))

        image_sizes = [(image_sizes[0].item(), image_sizes[1].item())]
        original_image_sizes = [(original_image_sizes[0].item(), original_image_sizes[1].item())]
        secondary_image_size = torch.Size([int(secondary_image_size[0].item()), int(secondary_image_size[1].item())])
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # proposals, proposal_losses = self.rpn(features, targets, image_sizes, secondary_image_size)
        # detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
        
        proposals, proposal_losses = self.rpn(features, None, image_sizes, secondary_image_size)
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, None)
        detections = self.postprocess(detections, image_sizes, original_image_sizes)  

        # losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)

        # print("------- detec", detections)
        return detections[0]["boxes"], detections[0]["labels"], detections[0]["scores"]


        return self.eager_outputs({}, detections)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
        

    def unquantized_forward(self, unquantized_tensort, image_sizes, original_image_sizes, secondary_image_size):
        '''image sizes '''
        # targets is None
        print("just before running backbone forward SplittableResNet?")
        features = self.backbone(unquantized_tensort)

        print("unquantized tensort", unquantized_tensort)
        print("image_sizes", image_sizes)
        print("original_image_sizes", original_image_sizes)
        print("secondary_image_size", secondary_image_size)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        # proposals, proposal_losses = self.rpn(features, targets, image_sizes, secondary_image_size)
        # detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
        proposals, proposal_losses = self.rpn(features, None, image_sizes, secondary_image_size)
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, None)
        detections = self.postprocess(detections, image_sizes, original_image_sizes)  # type: ignore[operator]

        # losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)

        return self.eager_outputs({}, detections)
        
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Lambda
#from VideoDataset import VideoDataset
#from VideoTransforms import *
import matplotlib.pyplot as plt
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)
import os
from collections import OrderedDict
from pytorchvideo.data import (
    Charades,
    LabeledVideoDataset,
    SSv2,
    make_clip_sampler,
)
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.transforms import (ApplyTransformToKey,ShortSideScale,UniformTemporalSubsample)
import numpy as np
import torchvision.utils as vutils
import av
import math
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import pandas as pd
from typing import Callable, List, Tuple, Union
import torch.nn.functional as F

from slowfast.models.build import MODEL_REGISTRY
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.ae_helper import create_encoder_head, create_res_decoder_stem,create_ae_res_stage, Pad3d, create_resnet_encoder
from slowfast.models.video_model_builder import _POOL1, _TEMPORAL_KERNEL_BASIS
from slowfast.models.ptv_model_builder import PTVResNetAutoencoder, PTVResNet
from functools import partial
from pytorchvideo.models.resnet import create_bottleneck_block,create_resnet
from pytorchvideo.models.head import create_res_basic_head, create_res_roi_pooling_head
from pytorchvideo.models.stem import (
    create_res_basic_stem,
)
from pytorchvideo.models.net import Net

from pytorchvideo.models.resnet import create_bottleneck_block, create_res_block, ResStage
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from slowfast.datasets.loader import construct_loader
from slowfast.models.ptv_model_builder import PTVResNet
from slowfast.datasets.ptv_datasets import Ptvfishbase
from VideoDataset import VideoDataset
from VideoTransforms import *

_MODEL_STAGE_DEPTH = {18:(1,1,1,1),50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
class Args:
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.shard_id = 0
        self.num_shards = 1
        self.init_method = 'tcp://localhost:9999'
        self.opts = None



def create_resnet_encoder(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (3, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    stem_pool: Callable = nn.MaxPool3d,
    stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
    stem_pool_stride: Tuple[int] = (1, 2, 2),
    stem: Callable = create_res_basic_stem,
    conv: Callable = nn.Conv3d,
    # Stage configs.
    stage1_pool: Callable = None,
    stage1_pool_kernel_size: Tuple[int] = (2, 1, 1),
    stage_conv_a_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 1, 1),
        (1, 1, 1),
        (3, 1, 1),
        (3, 1, 1),
    ),
    stage_conv_b_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
    ),
    stage_conv_b_num_groups: Tuple[int] = (1, 1, 1, 1),
    stage_conv_b_dilation: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    stage_spatial_h_stride: Tuple[int] = (1, 2, 2, 2),
    stage_spatial_w_stride: Tuple[int] = (1, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Union[Tuple[Callable], Callable] = create_bottleneck_block,
    # Head configs.
    head: Callable = create_res_basic_head,
    head_pool: Callable = nn.AvgPool3d,
    head_pool_kernel_size: Tuple[int] = (4, 7, 7),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = None,
    head_output_with_global_average: bool = True,
) -> nn.Module:
    """
    Build ResNet style models for video recognition. ResNet has three parts:
    Stem, Stages and Head. Stem is the first Convolution layer (Conv1) with an
    optional pooling layer. Stages are grouped residual blocks. There are usually
    multiple stages and each stage may include multiple residual blocks. Head
    may include pooling, dropout, a fully-connected layer and global spatial
    temporal averaging. The three parts are assembled in the following order:
    ::
                                         Input
                                           ↓
                                         Stem
                                           ↓
                                         Stage 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Stage N
                                           ↓
                                         Head
    Args:
        input_channel (int): number of channels for the input video clip.
        model_depth (int): the depth of the resnet. Options include: 50, 101, 152.
        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.
        norm (callable): a callable that constructs normalization layer.
        activation (callable): a callable that constructs activation layer.
        stem_dim_out (int): output channel size to stem.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.
        stem_pool (callable): a callable that constructs resnet head pooling layer.
        stem_pool_kernel_size (tuple): pooling kernel size(s).
        stem_pool_stride (tuple): pooling stride size(s).
        stem (callable): a callable that constructs stem layer.
            Examples include: create_res_video_stem.
        stage_conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        stage_conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        stage_conv_b_num_groups (tuple): number of groups for groupwise convolution
            for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
        stage_conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        stage_spatial_h_stride (tuple): the spatial height stride for each stage.
        stage_spatial_w_stride (tuple): the spatial width stride for each stage.
        stage_temporal_stride (tuple): the temporal stride for each stage.
        bottleneck (callable): a callable that constructs bottleneck block layer.
            Examples include: create_bottleneck_block.
        head (callable): a callable that constructs the resnet-style head.
            Ex: create_res_basic_head
        head_pool (callable): a callable that constructs resnet head pooling layer.
        head_pool_kernel_size (tuple): the pooling kernel size.
        head_output_size (tuple): the size of output tensor for head.
        head_activation (callable): a callable that constructs activation layer.
        head_output_with_global_average (bool): if True, perform global averaging on
            the head output.
    Returns:
        (nn.Module): basic resnet.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_resnet")

    # Given a model depth, get the number of blocks for each stage.
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    # Broadcast single element to tuple if given.
    if isinstance(stage_conv_a_kernel_size[0], int):
        stage_conv_a_kernel_size = (stage_conv_a_kernel_size,) * len(stage_depths)

    if isinstance(stage_conv_b_kernel_size[0], int):
        stage_conv_b_kernel_size = (stage_conv_b_kernel_size,) * len(stage_depths)

    if isinstance(stage_conv_b_dilation[0], int):
        stage_conv_b_dilation = (stage_conv_b_dilation,) * len(stage_depths)

    if isinstance(bottleneck, Callable):
        bottleneck = [
            bottleneck,
        ] * len(stage_depths)

    blocks = []
    # Create stem for resnet.
    stem = stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        conv=conv,
        pool=stem_pool,
        pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[size // 2 for size in stem_pool_kernel_size],
        norm=norm,
        activation=activation,
    )
    blocks.append(stem)

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4

    # Create each stage for resnet.
    for idx in range(len(stage_depths)):

        stage_dim_inner = stage_dim_out // 4
        depth = stage_depths[idx]

        stage_conv_a_kernel = stage_conv_a_kernel_size[idx]
        stage_conv_a_stride = (stage_temporal_stride[idx], 1, 1)
        stage_conv_a_padding = (
            [size // 2 for size in stage_conv_a_kernel]
            if isinstance(stage_conv_a_kernel[0], int)
            else [[size // 2 for size in sizes] for sizes in stage_conv_a_kernel]
        )

        stage_conv_b_stride = (
            1,
            stage_spatial_h_stride[idx],
            stage_spatial_w_stride[idx],
        )

        stage = create_ae_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=bottleneck[idx],
            conv_a_kernel_size=stage_conv_a_kernel,
            conv_a_stride=stage_conv_a_stride,
            conv_a_padding=stage_conv_a_padding,
            conv_a=conv,
            conv_b_kernel_size=stage_conv_b_kernel_size[idx],
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=(
                stage_conv_b_kernel_size[idx][0] // 2,
                stage_conv_b_dilation[idx][1]
                if stage_conv_b_dilation[idx][1] > 1
                else stage_conv_b_kernel_size[idx][1] // 2,
                stage_conv_b_dilation[idx][2]
                if stage_conv_b_dilation[idx][2] > 1
                else stage_conv_b_kernel_size[idx][2] // 2,
            ),
            conv_b_num_groups=stage_conv_b_num_groups[idx],
            conv_b_dilation=stage_conv_b_dilation[idx],
            conv_b=conv,
            conv_c=conv,
            conv_skip=conv,
            norm=norm,
            activation=activation,
        )

        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

        if idx == 0 and stage1_pool is not None:
            blocks.append(
                stage1_pool(
                    kernel_size=stage1_pool_kernel_size,
                    stride=stage1_pool_kernel_size,
                    padding=(0, 0, 0),
                )
            )
    if head is not None:
        head = head(
            pool=head_pool,
            output_size=head_output_size,
            pool_kernel_size=head_pool_kernel_size,
            activation=head_activation,
        )
        blocks.append(head)

    return Net(blocks=nn.ModuleList(blocks))

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=(1, 1, 1), upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        if stride[2] == 1:
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3),
                                   stride=stride, bias=False, padding=(0, 1, 1))
        else:
            if stride[0] != 1:
                padding = (1, 1, 1)
            else:
                padding = (0, 1, 1)
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels,
                                            kernel_size=(1, 3, 3),
                                            stride=stride, bias=False,
                                            padding=(0, 1, 1),
                                            output_padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out


class ResnetAutoencoder(nn.Module):
    DEPTH = 6

    def __init__(self, resnet, upblock, num_layers, n_classes, n_channels=3, ):
        super().__init__()
        self.in_channels = 256
        resnet = resnet  # torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        stem = resnet.blocks[0]
        self.input_block = nn.Sequential(*list(stem.children()))[:3]  # nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(stem.children())[3]  # list(resnet.children())[3]
        for bottleneck in list(resnet.blocks.children()):
            if isinstance(bottleneck, ResStage):  # isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.final_conv = nn.Conv3d(in_channels=256, out_channels=512,
                                    kernel_size=(8, 4, 4), stride=(4, 2, 2), padding=(2, 1, 1))
        # self.final_pool = nn.MaxPool3d(kernel_size=(8,4,4),stride=(4,2,2),padding=(2,0,0),return_indices=True)
        self.relu = nn.ReLU()
        self.initial_pool = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=(8, 4, 4),
                                               stride=(4, 2, 2), padding=(
            0, 0, 0))  # nn.MaxUnpool3d(kernel_size=(8,4,4),stride=(4,2,2),padding=(2,0,0))
        self.initial_conv = nn.ConvTranspose3d(in_channels=512, out_channels=256,
                                               kernel_size=(8, 4, 4), stride=(4, 2, 2), padding=(2, 1, 1))

        up_blocks.append(self._make_up_block(
            upblock, 128, num_layers[3], stride=(1, 2, 2)))
        up_blocks.append(self._make_up_block(
            upblock, 64, num_layers[2], stride=(1, 2, 2)))
        up_blocks.append(self._make_up_block(
            upblock, 32, num_layers[1], stride=(1, 2, 2)))
        up_blocks.append(self._make_up_block(
            upblock, 8, num_layers[0], stride=(1, 2, 2)))
        upsample = nn.Sequential(
            # nn.Upsample(scale_factor=(1,2,2)),
            # nn.Conv3d(self.in_channels,8, kernel_size=1, stride=1),
            nn.ConvTranspose3d(self.in_channels,  # 256
                               8,
                               kernel_size=1, stride=(1, 2, 2),
                               bias=False, output_padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
        )
        up_blocks.append(DeconvBottleneck(
            self.in_channels, 8, 1, (1, 2, 2), upsample))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv1_1 = nn.ConvTranspose3d(8, n_classes, kernel_size=1, stride=1,
                                          bias=False)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            if stride[1] != 1:
                if stride[0] != 1:
                    output_padding = (1, 1, 1)
                else:
                    output_padding = (0, 1, 1)

            else:
                output_padding = 0
            upsample = nn.Sequential(
                # nn.Upsample(scale_factor=(1,2,2)),
                # nn.Conv3d(self.in_channels, init_channels * 2, kernel_size=1, stride=1),
                nn.ConvTranspose3d(self.in_channels, init_channels * 2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=output_padding),
                nn.BatchNorm3d(init_channels * 2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(
            block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def encoder(self, x):
        pre_pools = {}
        pre_pools[f"layer_0"] = x.size()
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x.size()

        x = self.input_pool(x)
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (ResnetAutoencoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x.size()
        pre_pools['final'] = x.size()
        x = self.final_conv(x)
        # x,indices = self.final_pool(x)
        x = self.relu(x)

        x = F.avg_pool3d(x, kernel_size=(8, 4, 4))
        return x, pre_pools

    def decoder(self, x, pre_pools):
        x = self.initial_pool(x)
        x = self.initial_conv(x)
        x = self.relu(x)
        for i, block in enumerate(self.up_blocks, 1):
            # key = f"layer_{ResnetAutoencoder.DEPTH - 1 - i}"
            x = block(x)
        x = self.conv1_1(x, output_size=pre_pools[f"layer_0"])
        return x

    def forward(self, x):
        img = x
        z, pre_pools = self.encoder(x)

        recon = self.decoder(z, pre_pools)
        return z, recon


args = Args('/media/shirbar/DATA/codes/SlowFast/configs/FishBase/FAST_AUTOENCODER_32x2_R50.yaml')
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)


norm_module = get_norm(cfg)
#head_act = get_head_act(cfg.MODEL.HEAD_ACT)
pool_size = _POOL1[cfg.MODEL.ARCH]
num_groups = cfg.RESNET.NUM_GROUPS
spatial_dilations = cfg.RESNET.SPATIAL_DILATIONS
spatial_strides = cfg.RESNET.SPATIAL_STRIDES
temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
stage1_pool = pool_size[0][0] != 1 or len(set(pool_size[0])) > 1
stage_spatial_stride = (
            spatial_strides[0][0],
            spatial_strides[1][0],
            spatial_strides[2][0],
            spatial_strides[3][0],
        )
stage_conv_a_kernel_size = (
                (3, 1, 1),
                (3, 1, 1),
                (3, 1, 1),
                (3, 1, 1),
            )

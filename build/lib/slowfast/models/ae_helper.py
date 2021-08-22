from typing import Callable, List, Tuple, Union
from torch.autograd import Function, Variable
from torch.nn.modules.utils import _ntuple
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.head import create_res_basic_head, create_res_roi_pooling_head
from pytorchvideo.models.net import Net
from pytorchvideo.models.stem import (
    create_res_basic_stem,
)
from pytorchvideo.models.resnet import create_bottleneck_block, create_res_stage, create_res_block, ResStage

_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
_MODEL_STAGE_DEPTH = {18: (1, 1, 1, 1), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}

def create_encoder_head(
    *,
    # Pooling configs.
    pool: Callable = nn.AvgPool3d,
    output_size: Tuple[int] = (1, 1, 1),
    pool_kernel_size: Tuple[int] = (1, 7, 7),
    pool_stride: Tuple[int] = (1, 1, 1),
    pool_padding: Tuple[int] = (0, 0, 0),
    # Activation configs.
    activation: Callable = None,
    projection: Callable = nn.Linear,
) -> nn.Module:
    """
    Creates ResNet basic head. This layer performs an optional pooling operation
    followed by an optional dropout, a fully-connected projection, an activation layer
    and a global spatiotemporal averaging.
    ::
                                        Pooling
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging
    Activation examples include: ReLU, Softmax, Sigmoid, and None.
    Pool3d examples include: AvgPool3d, MaxPool3d, AdaptiveAvgPool3d, and None.
    Args:
        in_features: input channel size of the resnet head.
        out_features: output channel size of the resnet head.
        pool (callable): a callable that constructs resnet head pooling layer,
            examples include: nn.AvgPool3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d, and
            None (not applying pooling).
        pool_kernel_size (tuple): pooling kernel size(s) when not using adaptive
            pooling.
        pool_stride (tuple): pooling stride size(s) when not using adaptive pooling.
        pool_padding (tuple): pooling padding size(s) when not using adaptive
            pooling.
        output_size (tuple): spatial temporal output size when using adaptive
            pooling.
        activation (callable): a callable that constructs resnet head activation
            layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not
            applying activation).
        dropout_rate (float): dropout rate.
        output_with_global_average (bool): if True, perform global averaging on temporal
            and spatial dimensions and reshape output to batch_size x out_features.
    """

    if activation is None:
        activation_model = None
    elif activation == nn.Softmax:
        activation_model = activation(dim=1)
    else:
        activation_model = activation()

    if pool is None:
        pool_model = None
    elif pool == nn.AdaptiveAvgPool3d:
        pool_model = pool(output_size)
    else:
        pool_model = pool(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )

    return PoolAEHead(
        activation=activation_model,
        pool=pool_model,
    )

class PoolAEHead(nn.Module):
    """
    Pool the autoencoder bottleneck This layer performs a pooling operation an optional activation layer


    """

    def __init__(
        self,
        pool: nn.Module,
        activation: nn.Module = None,
    ) -> None:
        """
        Args:
            pool (torch.nn.modules): pooling module.
            activation (torch.nn.modules): activation module.
            output_pool (torch.nn.Module): pooling module for output.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Performs pooling.
        x = self.pool(x)
        # Performs activation.
        if self.activation is not None:
            x = self.activation(x)

        return x


def create_res_decoder_stem(
    *,
    # Conv configs.
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (3, 7, 7),
    conv_stride: Tuple[int] = (1, 2, 2),
    conv_padding: Tuple[int] = (1, 3, 3),
    conv_bias: bool = False,
    conv: Callable = nn.Conv3d,
    # Pool configs.
    pool: Callable = nn.MaxPool3d,
    pool_kernel_size: Tuple[int] = (1, 3, 3),
    pool_stride: Tuple[int] = (1, 2, 2),
    pool_padding: Tuple[int] = (0, 1, 1),
    pad3d: None,#Tuple[int] = (0,1,0,1,0,0),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:
    """
    Creates the basic resnet stem layer. It performs spatiotemporal Convolution, BN, and
    Relu following by a spatiotemporal pooling.
    ::
                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d
    Normalization options include: BatchNorm3d and None (no normalization).
    Activation options include: ReLU, Softmax, Sigmoid, and None (no activation).
    Pool3d options include: AvgPool3d, MaxPool3d, and None (no pooling).
    Args:
        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s).
        conv_padding (tuple): convolutional padding size(s).
        conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
            output.
        conv (callable): Callable used to build the convolution layer.
        pool (callable): a callable that constructs pooling layer, options include:
            nn.AvgPool3d, nn.MaxPool3d, and None (not performing pooling).
        pool_kernel_size (tuple): pooling kernel size(s).
        pool_stride (tuple): pooling stride size(s).
        pool_padding (tuple): pooling padding size(s).
        norm (callable): a callable that constructs normalization layer, options
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        activation (callable): a callable that constructs activation layer, options
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
    Returns:
        (nn.Module): resnet basic stem layer.
    """
    conv_module = conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        bias=conv_bias,

    )
    norm_module = (
        None
        if norm is None
        else norm(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = None if activation is None else activation()
    if pool == nn.ConvTranspose3d:
        pool_module = pool(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )
    elif pool == nn.Upsample:
        pool_module = pool(size=pool_kernel_size)
    else:
        pool_module = None
    if pad3d is not None:
        pad3d_module = Pad3d(pad3d)
    else:
        pad3d_module = None

    return ResNetDecoderStem(
        conv=conv_module,
        norm=norm_module,
        activation=activation_module,
        unpool=pool_module,
        pad3d=pad3d_module,
    )

class ResNetDecoderStem(nn.Module):
    """
    ResNet basic 3D stem module. Performs spatiotemporal Convolution, BN, and activation
    following by a spatiotemporal pooling.
    ::
                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d
    The builder can be found in `create_res_basic_stem`.
    """

    def __init__(
        self,
        *,
        conv: nn.Module = None,
        norm: nn.Module = None,
        activation: nn.Module = None,
        unpool: nn.Module = None,
        pad3d: nn.Module = None
    ) -> None:
        """
        Args:
            conv (torch.nn.modules): convolutional module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.unpool is not None:
            x = self.unpool(x)
        if self.pad3d is not None:
            x = self.pad3d(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Pad3d(nn.Module):
    """Wrapper for ReflectionPadNd function in 3 dimensions."""
    def __init__(self, padding: Union[int, Tuple[int]], mode: str = 'replicate'):
        super(Pad3d, self).__init__()
        if type(padding) == int:
            self.padding = _ntuple(6)(padding)
        elif type(padding) == tuple:
            assert len(padding) == 6
            self.padding = padding
        else:
            raise ValueError('input must be either integer or tuple')
        assert mode in ['constant', 'reflect', 'replicate']
        self.mode = mode

    def forward(self, x: Variable) -> Variable:
        return F.pad(x, self.padding, mode=self.mode)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


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

def create_resnet_decoder(
        *,
        # Input clip configs.
        input_channel: int = 3,
        # Model configs.
        model_depth: int = 50,
        model_num_class: int = 400,
        dropout_rate: float = 0.5,
        z_dim: int = 2048,
        # Normalization configs.
        norm: Callable = nn.BatchNorm3d,
        # Activation configs.
        activation: Callable = nn.ReLU,
        # Stem configs.
        stem_dim_out: int = 64,
        stem_conv_kernel_size: Tuple[int] = (3, 6, 6),
        stem_conv_stride: Tuple[int] = (1, 2, 2),
        stem_conv_padding: Tuple[int] = (2, 2, 2),
        stem_pool: Callable = nn.MaxPool3d,
        stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
        stem_pool_stride: Tuple[int] = (1, 2, 2),
        stem: Callable = create_res_decoder_stem,
        conv: Callable = nn.Conv3d,

        # Stage configs.
        stage1_pool: Callable = None,
        stage1_pool_kernel_size: Tuple[int] = (2, 1, 1),
        stage_conv_a_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
                (3, 1, 1),
                (3, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
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
    Build ResNet style Decoder for Video Autoencoder. ResNet has three parts:
    Stem, Stages and Head. Stem is the first Convolution layer (Conv1) with an
    optional pooling layer. Stages are grouped residual blocks. There are usually
    multiple stages and each stage may include multiple residual blocks. Head
    may include pooling, dropout, a fully-connected layer and global spatial
    temporal averaging. The three parts are assembled in the following order:
    ::
                                         Input
                                           ↓
                                         Initial Conv
                                           ↓
                                         Stage 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Stage N
                                           ↓
                                         Stem
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
    add_padding = False
    pad3d_padding = [(1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0)]
    # Create stem for resnet.
    initial_conv = nn.Sequential(nn.ConvTranspose3d(in_channels=z_dim, out_channels=z_dim,
                                                    kernel_size=(32, 8, 8), stride=(1, 1, 1), padding=(0, 0, 0)),
                                 nn.ReLU())
    blocks.append(initial_conv)

    stage_dim_in = z_dim

    stage_dim_out = stage_dim_in // 2

    # Create each stage for resnet.
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_in // 4
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
            add_pad3d=add_padding,
            pad3d_padding=pad3d_padding[idx % 2],
        )

        blocks.append(stage)
        stage_dim_in = stage_dim_out
        if idx == (len(stage_depths) - 2):
            stage_dim_out = stage_dim_out // 4
            add_padding = False
        else:
            stage_dim_out = stage_dim_out // 2
            add_padding = False


    stem = stem(
        in_channels=stem_dim_out,
        out_channels=input_channel,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=stem_conv_padding,
        conv=conv,
        pool=stem_pool,
        pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[size // 2 for size in stem_pool_kernel_size],
        norm=norm,
        activation=activation,
        pad3d=None,
    )
    blocks.append(stem)
    return Net(blocks=nn.ModuleList(blocks))


def create_ae_res_stage(
    *,
    # Stage configs.
    depth: int,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable,

    # Conv configs.
    conv_a_kernel_size: Union[Tuple[int], List[Tuple[int]]] = (3, 1, 1),
    conv_a_stride: Tuple[int] = (2, 1, 1),
    conv_a_padding: Union[Tuple[int], List[Tuple[int]]] = (1, 0, 0),
    conv_a: Callable = nn.Conv3d,
    conv_b_kernel_size: Tuple[int] = (1, 3, 3),
    conv_b_stride: Tuple[int] = (1, 2, 2),
    conv_b_padding: Tuple[int] = (0, 1, 1),
    conv_b_num_groups: int = 1,
    conv_b_dilation: Tuple[int] = (1, 1, 1),
    conv_b: Callable = nn.Conv3d,
    conv_c: Callable = nn.Conv3d,
    conv_skip: Callable = nn.Conv3d,
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
    add_pad3d: bool = False,
    pad3d_padding: Tuple[int] = (1,0,1,0,0,0),

) -> nn.Module:
    """
    Create Residual Stage, which composes sequential blocks that make up a ResNet. These
    blocks could be, for example, Residual blocks, Non-Local layers, or
    Squeeze-Excitation layers.
    ::
                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock
    Normalization examples include: BatchNorm3d and None (no normalization).
    Activation examples include: ReLU, Softmax, Sigmoid, and None (no activation).
    Bottleneck examples include: create_bottleneck_block.
    Args:
        depth (init): number of blocks to create.
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable that constructs bottleneck block layer.
            Examples include: create_bottleneck_block.
        conv_a_kernel_size (tuple or list of tuple): convolutional kernel size(s)
            for conv_a. If conv_a_kernel_size is a tuple, use it for all blocks in
            the stage. If conv_a_kernel_size is a list of tuple, the kernel sizes
            will be repeated until having same length of depth in the stage. For
            example, for conv_a_kernel_size = [(3, 1, 1), (1, 1, 1)], the kernel
            size for the first 6 blocks would be [(3, 1, 1), (1, 1, 1), (3, 1, 1),
            (1, 1, 1), (3, 1, 1)].
        conv_a_stride (tuple): convolutional stride size(s) for conv_a.
        conv_a_padding (tuple or list of tuple): convolutional padding(s) for
            conv_a. If conv_a_padding is a tuple, use it for all blocks in
            the stage. If conv_a_padding is a list of tuple, the padding sizes
            will be repeated until having same length of depth in the stage.
        conv_a (callable): a callable that constructs the conv_a conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_b_stride (tuple): convolutional stride size(s) for conv_b.
        conv_b_padding (tuple): convolutional padding(s) for conv_b.
        conv_b_num_groups (int): number of groups for groupwise convolution for
            conv_b.
        conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        conv_b (callable): a callable that constructs the conv_b conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_c (callable): a callable that constructs the conv_c conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        norm (callable): a callable that constructs normalization layer. Examples
            include nn.BatchNorm3d, and None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        activation (callable): a callable that constructs activation layer. Examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
    Returns:
        (nn.Module): resnet basic stage layer.
    """
    res_blocks = []
    if isinstance(conv_a_kernel_size[0], int):
        conv_a_kernel_size = [conv_a_kernel_size]
    if isinstance(conv_a_padding[0], int):
        conv_a_padding = [conv_a_padding]
    # Repeat conv_a kernels until having same length of depth in the stage.
    conv_a_kernel_size = (conv_a_kernel_size * depth)[:depth]
    conv_a_padding = (conv_a_padding * depth)[:depth]

    for ind in range(depth):
        block = create_res_block(
            dim_in=dim_in if ind == 0 else dim_out,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=conv_a_kernel_size[ind],
            conv_a_stride=conv_a_stride if ind == 0 else (1, 1, 1),
            conv_a_padding=conv_a_padding[ind],
            conv_a=conv_a,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride if ind == 0 else (1, 1, 1),
            conv_b_padding=conv_b_padding,
            conv_b_num_groups=conv_b_num_groups,
            conv_b_dilation=conv_b_dilation,
            conv_b=conv_b,
            conv_c=conv_c,
            conv_skip=conv_skip,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation_bottleneck=activation,
            activation_block=activation,
        )
        res_blocks.append(block)
    if add_pad3d:
        res_blocks.append(Pad3d(padding=pad3d_padding))
    return ResStage(res_blocks=nn.ModuleList(res_blocks))


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


class ResNetAutoencoder(nn.Module):
    DEPTH = 6

    def __init__(self, resnet, num_layers, upblock=DeconvBottleneck,  n_channels=3, ):
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
        self.conv1_1 = nn.ConvTranspose3d(8, n_channels, kernel_size=1, stride=1,
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
            if i == (ResNetAutoencoder.DEPTH - 1):
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

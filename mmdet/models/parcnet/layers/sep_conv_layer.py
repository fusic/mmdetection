from torch import Tensor
from typing import Optional, Tuple
from ..misc.profiler import module_profile
from .base_layer import BaseLayer
from .conv_layer import ConvLayer


class SeparableConv(BaseLayer):
    """
    This layer defines Depth-wise separable convolution, introduced in Xception
        https://arxiv.org/abs/1610.02357
    
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int or tuple,
                 stride: Optional[int or tuple] = 1,
                 dilation: Optional[int or tuple] = 1,
                 use_norm: Optional[bool] = True,
                 use_act: Optional[bool] = True,
                 bias: Optional[bool] = False,
                 padding_mode: Optional[str] = 'zeros',
                 opts_model_activation_name: str = "prelu",
                 opts_model_activation_neg_slope: float = 0.1,
                 opts_model_activation_inplace: bool = False,
                 opts_model_normalization_name: str = "batch_norm",
                 opts_model_normalization_groups: int = 1,
                 opts_model_normalization_momentum: float = 0.1,
                 *args, **kwargs):
        super(SeparableConv, self).__init__()
        self.dw_conv = ConvLayer(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=stride, dilation=dilation, groups=in_channels, bias=False, padding_mode=padding_mode,
            use_norm=True, use_act=False, opts_model_activation_name=opts_model_activation_name,
            opts_model_activation_neg_slope=opts_model_activation_neg_slope,
            opts_model_activation_inplace=opts_model_activation_inplace,
            opts_model_normalization_name=opts_model_normalization_name,
            opts_model_normalization_groups=opts_model_normalization_groups,
            opts_model_normalization_momentum=opts_model_normalization_momentum)
        self.pw_conv = ConvLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1,
            stride=1, dilation=1, groups=1, bias=bias, padding_mode=padding_mode,
            use_norm=use_norm, use_act=use_act, opts_model_activation_name=opts_model_activation_name,
            opts_model_activation_neg_slope=opts_model_activation_neg_slope,
            opts_model_activation_inplace=opts_model_activation_inplace,
            opts_model_normalization_name=opts_model_normalization_name,
            opts_model_normalization_groups=opts_model_normalization_groups,
            opts_model_normalization_momentum=opts_model_normalization_momentum
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def __repr__(self):
        repr_str = "{}(in_channels={}, out_channels={}, kernel_size={}, stride={}, dilation={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.dilation
        )
        return repr_str

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        params, macs = 0.0, 0.0
        input, p, m = module_profile(module=self.dw_conv, x=input)
        params += p
        macs += m

        input, p, m = module_profile(module=self.pw_conv, x=input)
        params += p
        macs += m

        return input, params, macs
import mxnet as mx 
import mxnet.gluon.nn as nn 
from mxnet.gluon.nn import HybridBlock


class Conv2dModule(HybridBlock):
    """A convolution layer with normalization and activation.

    Parameters
    ----------
    in_channel: int
        The input-dim of conv-layer.
    out_channel: int
        The output-dim of conv-layer.
    kernel_size: int (default 3)
        Size of kernel.
    padding: int (default 1)
        Padding size while convoluting. 
    strides: int (default 1)
        The stiding size while convoluting.
    with_bias: bool (default False)
        Whether to employ bias follow convolution.
    num_group: int (default 1)
        The number of convlution group.
    norm_cfg: dict
        Config of normalization.
    activation: str (default 'relu')
        Activation type.
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 padding=1,
                 strides=1,
                 dilation=1,
                 with_bias=False,
                 num_group=1,
                 norm_cfg=dict(
                     type='BatchNorm',
                     momentum=0.9, 
                     epsilon=1e-5),
                 activation='relu'):
        super(Conv2dModule, self).__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._kernel_size = kernel_size
        self._padding = padding
        self._strides = strides
        self._dilation = dilation
        self._with_bias = with_bias 
        self._num_group = num_group 
        self._norm_cfg = norm_cfg 
        self._ac_type = activation

        self._build_layers()


    def _build_layers(self):
        self.conv = nn.Conv2D(in_channels=self._in_channel,
                              channels=self._out_channel,
                              kernel_size=self._kernel_size,
                              strides=self._strides,
                              dilation=self._dilation,
                              use_bias=self._with_bias)
        if self._norm_cfg and isinstance(self._norm_cfg, dict):
            norm_cfg = self._norm_cfg.copy()
            norm_type = norm_cfg.pop('type')
            assert norm_type in ['BatchNorm', 'GroupNorm'] 
            self.norm = getattr(nn, norm_type)(**norm_cfg)
        else:
            self.norm = None 

        if self._ac_type:
            assert self._ac_type in ['relu', 'sigmoid', 'softrelu', 'softsign', 'tanh']

    def hybrid_forward(self, F, input):
        output = self.conv(input)
        if self.norm:
            output = self.norm(output)
        if self._ac_type:
            output = F.Activation(output, act_type=self._ac_type)

        return output 
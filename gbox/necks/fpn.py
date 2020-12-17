import mxnet as mx 
import mxnet.gluon.nn as nn 

from mxnet.gluon.nn import HybridBlock, HybridSequential

from ._necks import NECKS1  

#TODO recode interpolate


@NECKS.register()
class FPN(HybridBlock):
    r"""Feature Pyramid Network.
    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.
    Args:
        in_channels : List[int] 
            Number of input channels per scale.
        out_channels : int 
            Number of output channels (used at each scale)
        num_outs : int 
            Number of output scales.
        start_level : int (default: 0)
            Index of the start input backbone level used to
            build the feature pyramid. 
        end_level : int (default: -1)
            Index of the end input backbone level (exclusive) to
            build the feature pyramid. Which means the last level.
        add_extra_convs : bool or str 
            If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
    """

    def __init__(self,
                 in_channels, 
                 out_channels, 
                 num_outs, 
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = HybridSequential()
        self.fpn_convs = HybridSequential()

        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.add(
                HybridSequential(
                    nn.Conv2D(in_channels=in_channels[i],
                              channels=out_channels,
                              kernel_size=1,
                              padding=0),
                    nn.BatchNorm(),
                    nn.Activation('relu')
                )
            )

            self.fpn_convs.add(
                HybridSequential(
                    nn.Conv2D(in_channels=out_channels,
                              channels=out_channels,
                              kernel_size=3,
                              padding=1),
                    nn.BatchNorm(),
                    nn.Activation('relu')
                )
            )

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                self.fpn_convs.add(
                    HybridSequential(
                        nn.Conv2D(
                            in_channels=in_channels,
                            channels=out_channels,
                            strides=2,
                            kernel_size=3,
                            padding=1
                        ),
                        nn.BatchNorm(),
                        nn.Activation('relu')
                    )
                )

        def hybrid_forward(self, F, inputs):
            assert len(inputs) == len(self.in_channels)

            # build laterals
            laterals = [
                lateral_conv(inputs[i + self.start_level])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]

            used_backbone_levels = len(laterals)
            for i in range(used_backbone_levels - 1, 0, -1):
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.contrib.BilinearResize2D(laterals[i], like=laterals[i - 1])

            # build outputs
            # part 1: from original levels
            outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]

            if self.num_outs > len(outs):
                # use max pool to get more levels on top of outputs
                # (e.g., Faster R-CNN, Mask R-CNN)
                if not self.add_extra_convs:
                    for i in range(self.num_outs - used_backbone_levels):
                        outs.append(F.contrib.BilinearResize2D(out[-1], scale_heigh=0.5, scale_width=0.5))
                else:
                    if self.add_extra_convs == 'on_input':
                        extra_source = inputs[self.backbone_end_level - 1]
                    elif self.add_extra_convs == 'on_lateral':
                        extra_source = laterals[-1]
                    elif self.add_extra_convs == 'on_output':
                        extra_source = outs[-1]
                    else:
                        raise NotImplementedError
                    outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                    for i in range(used_backbone_levels + 1, self.num_outs):
                        if self.relu_before_extra_convs:
                            outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                        else:
                            outs.append(self.fpn_convs[i](outs[-1]))
            
            return tuple(outs)
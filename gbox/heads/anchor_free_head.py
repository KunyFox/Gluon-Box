import sys 
import mxnet as mx 
import mxnet.gluon.nn as nn 

sys.path.append('.')
from utils import func_mapping
from gbox.core import Conv2dModule
from base import BaseHead 
from mxnet.gluon.nn import HybridBlock, HybridSequential

from gbox import HEADS


@HEADS.register()
class AnchorFreeHead(BaseHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 num_layers=4,
                 feat_channels=256,
                 with_bias=False,
                 norm_cfg=dict(type='BatchNorm'),
                 feat_strides=(8, 16, 32, 64, 128),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss', 
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(AnchorFreeHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_layers = num_layers 
        self.with_bias = with_bias
        self.norm_cfg = norm_cfg
        self.feat_channels = feat_channels 
        self.feat_strides = feat_strides 
        self.loss_cls_cfg = loss_cls
        self.loss_bbox_cfg = loss_bbox 
        self.train_cfg = train_cfg 
        self.test_cfg = test_cfg 
        self.cls_out_channels = num_classes

        self.build_layers()


    def build_layers(self):
        self.build_cls_layers()
        self.build_reg_layers()
        self.build_pre_layers()


    def build_cls_layers(self):
        cls_layers = HybridSequential()
        for i in range(self.num_layers):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_layers.add(
                Conv2dModule(
                    in_channel=chn, 
                    out_channel=self.feat_channels,
                    kernel_size=3, 
                    padding=1, 
                    strides=1,
                    with_bias=self.with_bias,
                    norm_cfg=self.norm_cfg
                )
            )

        self.cls_layers = cls_layers 


    def build_reg_layers(self):
        reg_layers = HybridSequential()
        for i in range(self.num_layers):
            chn = self.in_channels if i == 0 else self.feat_channels
            reg_layers.add(
                Conv2dModule(
                    in_channel=chn, 
                    out_channel=self.feat_channels,
                    kernel_size=3, 
                    padding=1, 
                    strides=1,
                    with_bias=self.with_bias,
                    norm_cfg=self.norm_cfg
                )
            )

        self.reg_layers = reg_layers 

    
    def build_pre_layers(self):
        self.cls_pre = nn.Conv2D(in_channels=self.feat_channels, channels=self.cls_out_channels, \
                                 kernel_size=3, padding=1)
        self.reg_pre = nn.Conv2D(in_channels=self.feat_channels, channels=4, \
                                 kernel_size=3, padding=1)

    def hybrid_forward(self, F, inputs, **kwargs):
        if isinstance(inputs, mx.nd.NDArray):
            inputs = [inputs]
        assert isinstance(inputs, list)
        
        return func_mapping(self.base_hybrid_forward, inputs, F=F)


    def base_hybrid_forward(self, inputs, F, **kwargs):
        cls_feats = self.cls_layers(inputs)
        cls_preds = self.cls_pre(cls_feats)

        reg_feats = self.reg_layers(inputs)
        reg_preds = self.reg_pre(reg_feats)

        if kwargs.get('return_feat', False):
            return cls_preds, reg_preds, cls_feats, reg_feats
        return cls_preds, reg_preds


    def loss(self, cls_preds, reg_preds, labels, bboxes, **kwargs):
        """Compute loss of 'bbox' and 'classification'.

        Parameters:
        -----------
        cls_preds: list of NDArray
            Prdiction of classification.
        reg_preds: list of NDArray
            Prediction of regression.
        labels: list of NDArray
            The labels of bboxes.
        bboxes: list of NDArray
            The coordinates of booxes.
        """
        raise NotImplementedError


    def get_bboxes(self, cls_preds, reg_preds, img_infos):
        """Mapping outputs to bboxes.
        
        """
        raise NotImplementedError

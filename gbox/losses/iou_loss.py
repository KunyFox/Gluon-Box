import mxnet as mx 

from mxnet.gluon.nn import HybridBlock 
from ._losses import LOSSES



@LOSSES.register()
class IOULoss(HybridBlock):
    def __init__(self,
                 weight=None,
                 batch_axis=0,
                 mode='iou',
                 **kwargs):
        super(IOULoss, self).__init__()
        self._weight = weight 
        self._batch_axis = batch_axis 
        self._mode = mode 


    def hybrid_forward(self, F, bbox_preds, bbox_gt, esp=1e-16):
        assert bbox_preds.shape[-1] == 4 or bbox_preds.shape[0] == 0
        assert bbox_gt.shape[-1] == 4 or bbox_gt.shape[0] == 0
        assert bbox_preds.shape == bbox_gt.shape

        area_preds = (bbox_preds[..., 2] - bbox_preds[..., 0]) * \
                     (bbox_preds[..., 3] - bbox_preds[..., 1])
        area_gt = (bbox_gt[..., 2] - bbox_gt[..., 0]) * \
                  (bbox_gt[..., 3] - bbox_gt[..., 1])
        
        left_top = mx.nd.maximum(bbox_preds[..., :2], bbox_gt[..., :2])
        right_bottom = mx.nd.minimum(bbox_preds[..., 2:], bbox_gt[..., 2:])

        shape = (right_bottom - left_top).clip(a_min=0.0, a_max=1e12)
        overlap = shape[..., 0] * shape[..., 1]
        union = area_preds + area_gt - overlap 

        union = mx.nd.maximum(union, esp)
        ious = overlap / union
        if self._mode == 'giou':
            enclosed_lt = mx.nd.minimum(bbox_preds[..., :2], bbox_gt[..., :2])
            enclosed_rb = mx.nd.maximum(bbox_preds[..., 2:], bbox_gt[..., 2:])
            enclosed_shape = (enclosed_rb - enclosed_lt).clip(a_min=0.0, a_max=1e12)
            enclosed_area = enclosed_shape[..., 0] * enclosed_shape[..., 1]
            enclosed_area = mx.nd.maximum(enclosed_area, esp)
            ious = ious - (enclosed_area - union) / enclosed_area
        
        return ious

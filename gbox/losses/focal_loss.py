import mxnet as mx 

from mxnet.gluon.nn import HybridBlock 
from ._losses import LOSSES

EPS = 1e-12

@LOSSES.register()
class FocalLoss(HybridBlock):
    """Focal Loss <https://arxiv.org/abs/1708.02002>

    Implement of Focal Loss by referencing to gluoncv.
    
    """
    def __init__(self,
                 num_classes,
                 weight=None,
                 gamma=2.0,
                 alpha=0.25,
                 batch_axis=0,
                 from_logits=False,
                 sparse_label=True,
                 **kwargs):
        super(FocalLoss, self).__init__()
        self._num_classes = num_classes
        self._weight = weight 
        self._gamma = gamma 
        self._alpha = alpha 
        self._batch_axis = batch_axis
        self._from_logits = from_logits 
        self._sparse_label = sparse_label 


    def hybrid_forward(self, F, inputs, labels):
        if not self._from_logits:
            inputs = F.sigmoid(inputs)

        if self._sparse_label:
            labels = F.one_hot(labels, )
        else:
            labels = labels > 0

        pt = F.where(labels, inputs, 1 - inputs)
        t = F.ones_like(labels)
        alpha = F.where(labels, self._alpha * t, (1 - self._alpha) * t)
        loss = -alpha * ((1 - pt) ** self._gamma) * F.log(F.minimum(pt + EPS, 1))
        if self._weight:
            loss = self._weight * loss 

        return F.mean(loss, self._batch_axis, exclude=True)
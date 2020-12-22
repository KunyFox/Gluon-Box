import mxnet as mx 

from mxnet.gluon.nn import HybridBlock 
from ._losses import LOSSES



@LOSSES.register()
class CrossEntropyLoss(HybridBlock):
    def __init__(self, 
                 weight=None, 
                 logit_axis=-1, 
                 batch_axis=0,
                 sparse_label=True, 
                 from_logits=False, 
                 **kwargs):
        self._weight = weight 
        self._logit_axis = logit_axis 
        self._batch_axis = batch_axis 
        self._sparse_label = sparse_label 
        self._from_logits = from_logits 


    def hybrid_forward(self, F, inputs, labels):
        if not self._from_logits:
            inputs = F.log_softmax(inputs, self._logit_axis)
        
        if self._sparse_label:
            loss = -pick(pred, label, axis=self._axis, keepdims=True)
        else:
            labels = labels.reshape(inputs.shape)
            loss = -(inputs * labels).sum(axis=self._logit_axis, keepdims=True)
        if self._weight:
            loss = loss * self._weight
        
        return loss.mean(axis=self._batch_axis, exclude=True)
        



@LOSSES.register()
class BinaryCrossEntropyLoss(HybridBlock):
    def __init__(self,
                 weight=None,
                 from_logits=False,
                 batch_axis=0,
                 pos_weight=None,
                 **kwargs):
        super(BinaryCrossEntropyLoss, self).__init__()
        self._weight = weight 
        self._from_logits = from_logits 
        self._batch_axis = batch_axis 
        self._pos_weight = pos_weight


    def hybrid_forward(self, F, inputs, labels):
        labels = labels.reshape(inputs.shape)
        if not self._from_logits:
            if self._pos_weight:
                # x - x * z + (1 + z * pos_weight - z) * \
                #    (log(1 + exp(-abs(x))) + max(-x, 0))
                log_weight = 1 + F.broadcast_mul(self._pos_weight - 1 , labels)
                loss = inputs - inputs * labels + log_weight * \
                       (F.Activation(-F.abs(inputs), act_type='softrelu')) + \
                        F.relu(-inputs)
            else:
                # max(x, 0) - x * z + log(1 + exp(-abs(x)))
                loss = F.relu(inputs) - inputs * labels + \
                       F.Activation(-F.abs(inputs), act_type='softrelu')
        else:
            eps = 1e-12 
            if self._pos_weight:
                loss = -(F.broadcast_mul(F.log(inputs + eps) * labels, self._pos_weight)
                       + F.log(1. - inputs + eps) * (1. - labels))
            else:
                loss = -(F.log(iputs + eps) * labels
                       + F.log(1. - inputs + eps) * (1. - labels))
        if self._weight:
            loss = loss * self._weight 

        return F.mean(loss, axis=self._batch_axis, exclude=True)
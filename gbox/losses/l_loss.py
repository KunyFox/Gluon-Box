import mxnet as mx 

from mxnet.gluon.nn import HybridBlock 
from ._losses import LOSSES


@LOSSES.register()
class L1Loss(HybridBlock):
    def __init__(self, 
                 weight=None,
                 batch_axis=0):
        super(L1Loss, self).__init__()
        self._weight = weight 
        self._batch_axis = batch_axis 

    def hybrid_forward(self, F, inputs, labels):
        labels = labels.reshape(inputs.shape)
        loss = F.abs(labels - inputs)
        if self._weight:
            loss = loss * self._weight 

        return F.mean(loss, axis=self._batch_axis, exclude=True)



@LOSSES.register()
class L2Loss(L1Loss):
    def __init__(self, 
                 weight=None,
                 batch_axis=0):
        super(L2Loss, self).__init__(weight, batch_axis)
    
    def hybrid_forward(self, F, inputs, labels):
        labels = labels.reshape(inputs.shape)
        loss = F.square(labels - inputs)
        if self._weight:
            loss = loss * (self._weight / 2) 

        return F.mean(loss, axis=self._batch_axis, exclude=True)


@LOSSES.register()
class MSELoss(L2Loss):
    def __init__(self, 
                 weight=None,
                 batch_axis=0):
        super(MSELoss, self).__init__(weight, batch_axis)



@LOSSES.register()
class SmoothL1Loss(L1Loss):
    def __init__(self,
                 weight=None,
                 batch_axis=0,
                 beta=1.0):
        super(SmoothL1Loss, self).__init__(weight, batch_axis)
        self._beta = beta 

    def hybrid_forward(self, F, inputs, labels):
        labels = labels.reshape(inputs.shape) 
        diff = F.abs(labels - inputs) 
        loss = F.where(diff < self._beta, 0.5 * diff * diff / self._beta,
                       diff - 0.5 * self._beta) 
        return F.mean(loss, axis=self._batch_axis, exclude=True)

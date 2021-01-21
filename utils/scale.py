import mxnet 
import mxnet.gluon.nn as nn 


class Scale(nn.HybridBlock):
    def __init__(self, scale=1.0):
        self.scale = scale 

    def hybrid_forward(self, F, x):
        return x * self.scale 
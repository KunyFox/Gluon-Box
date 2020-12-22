import mxnet.gluon.nn as nn 

from abc import abstractmethod
from mxnet.gluon.nn import HybridBlock


class BaseHead(HybridBlock):
    def __init__(self):
        super(BaseHead, self).__init__()

    @abstractmethod
    def build_layers(self):
        pass

    @abstractmethod
    def loss(self, **kwargs):
        pass 

    @abstractmethod
    def get_bboxes(self, **kwargs):
        pass 

    @abstractmethod
    def train_bboxes(self, **kwargs):
        pass 

    def test_bboxes(self, F, feats, img_info, **kwargs):
        pass 
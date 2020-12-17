from mxnet.initializer import Xavier
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}



class SSD_VGG(HybridBlock):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each feature block.
    filters : list of int
        Numbers of filters in each feature block. List length should match the layers.
    classes : int, default 1000
        Number of classification classes.
    batch_norm : bool, default False
        Use batch normalization.
    """
    def __init__(self, num_layers, classes=1000, batch_norm=True, **kwargs):
        super(VGG, self).__init__(**kwargs)
        layers, filters = vgg_spec(num_layers)
        assert len(layers) == len(filters)
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm)
            self.features.add(nn.Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(nn.Dropout(rate=0.5))
            self.features.add(nn.Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(nn.Dropout(rate=0.5))
            self.output = nn.Dense(classes,
                                   weight_initializer='normal',
                                   bias_initializer='zeros')

    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x
import os
import mxnet as mx

root = "/home/ykun/DiskWD/Gluon-Box/backbone_models"

files = os.listdir(root)

for file in files:
    param = mx.nd.load(os.path.join(root, file))
    _ = param.pop('backbone.output.weight')
    _ = param.pop('backbone.output.bias')
    mx.nd.save('/home/ykun/DiskWD/Gluon-Box/model_zoo/{}'.format(file), param)
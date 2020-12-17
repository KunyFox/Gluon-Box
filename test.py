import gluoncv 

net = gluoncv.model_zoo.get_vgg(11, True, root='/home/ykun/DiskWD/Gluon-Box/backbone_models', batch_norm=True)

import pdb;pdb.set_trace()
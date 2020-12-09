# <font color=orange> Gluon-Box </font>
Introduction
------------
<font color=orange>Gluon-Box</font> is an open source object-detection tookit, which is implemented in [MxNet](https://mxnet.apache.org/) 1.6. The project is supported by CortexLabs.

![image](https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000080671.jpg)![image](https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000002157.jpg)

Supported [backbones](https://github.com/KyuanCortex/Gluon-Box/tree/main/gbox/backbones): 
- [ ] VGG
- [x] ResNetV1 
- [ ] ResNetV2 
- [ ] ResNeXt 
- [ ] RegNet
- [ ] HRNet 

Supported [necks](https://github.com/KyuanCortex/Gluon-Box/tree/main/gbox/necks):
- [ ] FPN

Supported Datasets:
- [x] [COCO](https://cocodataset.org/#home)

Supported [processers](https://github.com/KyuanCortex/Gluon-Box/blob/main/datasets/processer.py):
- [x] ImageReader 
- [x] ImageNormalizer
- [x] ImageResizer 
- [x] ImageFliper
- [x] ToBatch
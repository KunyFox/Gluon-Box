# <font color=orange> Gluon-Box </font>
Introduction
------------
<font color=orange>Gluon-Box</font> is an open source object-detection tookit, which is implemented in [MxNet](https://mxnet.apache.org/) 1.6. The project is supported by CortexLabs.

Supported backbones: 
- [ ] VGG
- [x] ResNetV1 
- [ ] ResNetV2 
- [ ] ResNeXt 
- [ ] RegNet
- [ ] HRNet 

Supported necks:
- [x] FPN

Supported Datasets:
- [x] [COCO](https://cocodataset.org/#home)

Supported [processers](https://github.com/KyuanCortex/Gluon-Box/blob/main/datasets/processer.py):
- [x] ImageReader 
- [x] ImageNormlation
- [x] ImageResize 
- [x] ImageFlip
- [x] ToBatch
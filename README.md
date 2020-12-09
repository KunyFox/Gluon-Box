# <font color=orange> Gluon-Box </font>
Introduction
------------
<font color=orange>Gluon-Box</font> is an open source object-detection tookit, which is implemented in [MxNet](https://mxnet.apache.org/) 1.6. The project is supported by CortexLabs.
<div  align="center"> 
<img src="https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000080671.jpg" width="350" height="250"/><img src="https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000002157.jpg" width="350" height="250"/> 
</div>

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
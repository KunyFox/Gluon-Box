# <font color=orange> Gluon-Box </font>
Introduction
------------
<font color=orange>Gluon-Box</font> is an open source object-detection tookit, which is implemented in [MxNet](https://mxnet.apache.org/) 1.6. The project is supported by CortexLabs.
<div  align="center"> 
<img src="https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000080671.jpg" width="350" height="250"/><img src="https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000002157.jpg" width="350" height="250"/> 
<img src="https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000010363.jpg" width="175" height="125"/><img src="https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000005992.jpg" width="175" height="125"/><img src="https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000118209.jpg" width="175" height="125"/><img src="https://github.com/KyuanCortex/Gluon-Box/blob/main/imgs/000000221693.jpg" width="175" height="125"/>
</div>

Supported [_backbones_](https://github.com/KyuanCortex/Gluon-Box/tree/main/gbox/backbones): 
- [ ] VGG
- [x] ResNetV1 
- [x] ResNetV2 
- [ ] ResNeXt 
- [ ] RegNet
- [ ] HRNet 

Supported [_necks_](https://github.com/KyuanCortex/Gluon-Box/tree/main/gbox/necks):
- [x] FPN

Supported _Datasets_:
- [x] [COCO](https://cocodataset.org/#home)
- [ ] [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [ ] [LVIS](https://www.lvisdataset.org/)
- [ ] [Open Image](https://storage.googleapis.com/openimages/web/index.html)

Supported [_processers_](https://github.com/KyuanCortex/Gluon-Box/blob/main/datasets/processer.py):
- [x] ImageReader 
- [x] ImageNormalizer
- [x] ImageResizer 
- [x] ImageFliper
- [x] ToBatch
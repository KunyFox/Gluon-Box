# <font color=orange> Gluon-Box </font>
<div  align="center"> 
<img src="https://github.com/KunyFox/Gluon-Box/blob/main/imgs/000000080671.jpg" width="350" height="250"/><img src="https://github.com/KunyFox/Gluon-Box/blob/main/imgs/000000002157.jpg" width="350" height="250"/> 
<img src="https://github.com/KunyFox/Gluon-Box/blob/main/imgs/000000010363.jpg" width="175" height="125"/><img src="https://github.com/KunyFox/Gluon-Box/blob/main/imgs/000000005992.jpg" width="175" height="125"/><img src="https://github.com/KunyFox/Gluon-Box/blob/main/imgs/000000118209.jpg" width="175" height="125"/><img src="https://github.com/KunyFox/Gluon-Box/blob/main/imgs/000000221693.jpg" width="175" height="125"/>
</div>

## Introduction
Gluon-Box is an open source object-detection tookit, which is based on [MxNet](https://mxnet.apache.org/). The project is supported by CortexLabs. \
Gluon-box is implemented by referencing to [MMDetection](https://github.com/open-mmlab/mmdetection).
## These components have already been supported.
| Supported Components |   |
| :----------- | :-----------|
| [Backbones](https://github.com/KunyFox/Gluon-Box/tree/main/gbox/backbones)| ResNetV1, ResNetV2 |
| [Necks](https://github.com/KunyFox/Gluon-Box/tree/main/gbox/necks) | FPN |
| [Datasets](https://github.com/KunyFox/Gluon-Box/tree/main/gbox/datasets) | COCO |
| [Processers](https://github.com/KunyFox/Gluon-Box/blob/main/datasets/processer.py) | ImageReader, ImageNormalizer, ImageResizer, ImageFliper, BboxTransformer, ToBatch |
| [Losses](https://github.com/KunyFox/Gluon-Box/tree/main/gbox/losses) | CrossEntropyLoss(CE), BinaryCrossEntropyLoss(BCE), L1, L2, SmoothL1, FocalLoss |

<br/>  

### And we are working hard to support more methods.
| Coming Soon |   |
| :-------- | :--------- | 
| backbones | VGG, ResNext, RegNet, HRNet |
| datasets | PASCAL VOC, LVIS, Open Image |
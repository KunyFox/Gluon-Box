# BACKBONES

Here, we implement some backbones which are pre-trained on ImageNet. The logit layer is removed from the top of architecture.

## ResNet  [\<arxiv>](https://arxiv.org/pdf/1512.03385.pdf)
We adopt the ResNet in MXNet [model-zoo](https://mxnet.cdn.apache.org/versions/1.7.0/api/python/docs/api/gluon/model_zoo/index.html), including ResNet-v1 and ResNet-v2. The outputs are feature maps of four stages, and the pre-trained model could be downloaded from:
- [ResNetv1-18](https://drive.google.com/file/d/1pQgVB5UzpuTqMlAUNWIt4nmJ98M8qrC4/view?usp=sharing)
- [ResNetv1-34](https://drive.google.com/file/d/1pQgVB5UzpuTqMlAUNWIt4nmJ98M8qrC4/view?usp=sharing)
- [ResNetv1-50](https://drive.google.com/file/d/1OlOtOx9NZhv9Ls3wZUtq-qLoCGwk5FXv/view?usp=sharing)
- [ResNetv1-101](https://drive.google.com/file/d/13OorURLnwLg6_J6znimKQ_n05ahOD6UI/view?usp=sharing)
- [ResNetv1-152](https://drive.google.com/file/d/13MSXempm4uISAGF7W41JFk3hw63W-4df/view?usp=sharing)
- [ResNetv2-18](https://drive.google.com/file/d/12XK75RJXa9v7DbBJoQWq6Ijn7kEjT-J-/view?usp=sharing)
- [ResNetv2-34](https://drive.google.com/file/d/1gD7_RQ_O9ZpMGMgZ4krhqXk_fMHGgJnY/view?usp=sharing)
- [ResNetv2-50](https://drive.google.com/file/d/11fetPrutSA1kuxnPLyy48fGNjKQc-FhA/view?usp=sharing)
- [ResNetv2-101](https://drive.google.com/file/d/1ITaspCRgsu6sISxRuiAnl7LxJ3s26P65/view?usp=sharing)
- [ResNetv2-162](https://drive.google.com/file/d/1KgjILc_KpsgQZouab72R5yR2AhqbOMab/view?usp=sharing)
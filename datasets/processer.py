# --------------------------------------------------------------------------------------
# Copyright 2020 by Kun Yuan, CortexLabs AI Group.
# All rights reserved.
# This file is part of the GLuon-Box (https://github.com/KyuanCortex/Gluon-Box),
# and is released under the "GNU General Public License v2.0". Please see the LICENSE.
# File that should have been included as part of this package.
#
# Processer is a function-lib for image reading and processing, the image and bboxes wi-
# ll be loaded as NDArray. Image information will be unpaked as a dict termed img_info.
# --------------------------------------------------------------------------------------


import os
import mxnet as mx 
import numpy as np 
from random import sample 
from gluoncv.data import imdecode
from gluoncv.data.transforms import image

from ._processer import PROCESSER



@PROCESSER.register()
class ImageReader(object):
    """Read image from file.

    Here we define image reading as a kind of processer.
    The function imdecode in gluoncv is employed as image
    reader.

    Parameters
    ----------
    color_type : str (default : 'RGB')
        The format of image.
        Optional: 'RGB', 'BGR'
    dtype : str (default : 'float32')
        The final dtype of image. 
        Optional: 'uint8', 'float32', 'float16'
    rescale : bool (default : True)
        Wether to rescale the value of pixel from [0, 255] to [0, 1].
        If dtype is 'uint8', rescale will be reset to False.
    """

    def __init__(self, 
                 color_type='RGB',
                 dtype='float32',
                 rescale=True):
        assert color_type in ['RGB', 'BGR']
        self.color_type = 1 if color_type=='RGB' else 0 
        
        self.dtype = dtype 
        assert dtype in ['uint8', 'float32', 'float16'],
            "{} is not supported now, optional dtype are 'uint8', 'float32' and 'float16'".format(dtype)
        if dtype is 'uint8':
            self.rescale = False 
        else:
            self.rescale = rescale

    def __call__(self, img_info):
        file_path = img_info['info']['file_name'] if img_info['info']['img_root'] is None  \
                    else os.path.join(img_info['info']['img_root'], img_info['info']['file_name'])
        with open(file_path, 'rb') as f:
            img = f.read()
            if self.
            img = imdecode(img, to_rgb=self.color_type) 
        
        if self.dtype is not 'uint8':
            img = mx.nd.Cast(img, dtype=self.dtype)

        if self.rescale:
            img = img / 255. 

        img_info.update({'img': img})

        return img_info
        
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'color_type={self.color_type}, '
                    f"dtype='{self.dtype}', "
                    f'rescale={self.rescale})')
        return repr_str


@PROCESSER.register()
class ImageNormalizer(object):
    """Normalize image with mean and std.

    Image shape should be (H x W x 3) or (B x 3 x H x W).
    The image should be a NDArray with 3 channels.

    Parameters
    ----------
    mean : tuple of float (default : (0, 0, 0))
        The mean of image channels.
    std : tuple of float (deafult : (1., 1., 1.))
        The std of image channels.
    format : str (default : 'HWC')
        The shape formation of image.
        Optional: 'HWC', 'BCHW'
    """
    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 format='HWC'):
        assert format in ['HWC', 'BCHW'] 
        self.format = format 
        self._mean = mx.nd.array(mean)
        self._std = 1 / mx.nd.array(std) 
        if format is 'HWC':
            self._mean = self._mean.reshape((1,1,3))
            self._std = self._std.reshape((1,1,3))
        else:
            self._mean = self._mean.reshape((1, 3, 1, 1))
            self._std = self._std.reshape((1, 3, 1, 1))

    def __call__(self, img_info):
        img = img_info.get('img', None)
        if img is None:
            raise TypeError('NoneType is not supported!')
        img = mx.nd.broadcast_sub(img, self._mean) 
        img = mx.nd.broadcast_mul(img, self._std)
        img_info['img'] = img 
        return imimg_info


@PROCESSER.register()
class ImageResizer(object):
    """Resize image and booxes.

    The image should be a NDArray shape of (H x W x 3).

    Parameters
    ----------
    scale : list of tuple
        The scales for resizing.
    muilt_mode: str (default : 'range')
        The sample mode of final scale when the length of scale 
        is larger than 1.
        Optional: 
            - 'range': random sample a value from scale[0] to scale[1].
            - 'fix': random sample a value between scale[0] and scale[1].
    keep_ratio : bool (default : True)
        Wether to keep aspect ratio of image.
    """
    def __init__(self, 
                 scale, 
                 muilt_mode='range',
                 keep_ratio=True):
        assert isinstance(scale, list)
        for s in scale:
            if not isinstance(s, tuple) or len(s) != 2:
                raise ValueError('Scale should be a list of tuple!')
        self.scale = scale 
        self.muilt_mode = muilt_mode 
        self.keep_ratio = keep_ratio 

    def __call__(self, img_info):
        scale = self._get_scale()
        img = img_info['img']
        bboxes = img_info['bboxes']
        h, w, _ = img.shape
        if not self.keep_ratio:
            scale_ratio = (scale[0] / h, scale[1] / w)
        else:
            scale_ratio = min(scale[0] / h, scale[1] / w)

        img_info['info'].updata({'scale_ratio': scale_ratio})
        img_info['img'] = self._resize_img(img, scale_ratio)
        img_info['bboxes'] = self._resize_bboxes(bboxes, scale_ratio)
        
        return img_info

    def _resize_img(self, img, ratio):
        h, w, _ = img.shape
        if isinstance(ratio, tuple):
            img = image.resize(img, w=scale[1], h=scale[0])
        else:
            img = image.resize(img, w=int(scale_ratio * w), h=int(scale_ratio * h))
        return img

    def _resize_bboxes(self, bboxes, ratio):
        if isinstance(ratio, tuple):
            ratio = np.array([ratio[1], ratio[0], ratio[1], ratio[0]]).reshape(1, 4)
        bboxes = bboxes * ratio 
        return bboxes 


    def _get_scale(self):
        if len(self.scale) == 1:
            return self.scale[0]

        if self.muilt_mode == 'fix':
            return sample(self.scale, 1)

        max_h = max([s[0] for s in self.scale])
        min_h = min([s[0] for s in self.scale])

        max_w = max([s[1] for s in self.scale])
        min_w = min([s[1] for s in self.scale])

        scale_h = np.random.randint(min_h, max_h + 1)
        scale_w = np.random.randint(min_w, max_w + 1)

        return (scale_h, scale2)


@PROCESSER.register()
class ImageFliper(object):
    """Flip the image and bboxes.

    If direction is 'h', the image will be flip along vertical axis.
    If direction is 'w', the image will be flip along horizontal axis.
    If direction is 'diag', the image will be flip along diagonal.

    Parameters
    ----------
    ratio : float or list of float (default : 0.5)
        The probability of flipping image.
    direction : str or list of str (default : 'h')
        The direction of flipping. If direction is a list of str, its length 
        must equal to ratio's. 
        Optional: 'h', 'w', 'diag'.

    """
    def __init__(self, 
                 ratio=0.5, 
                 direction='h'):
        if isinstance(ratio, float):
            assert 0 <= ratio <= 1. 
            self.ratio = [ratio]
        elif isinstance(ratio, list):
            for r in ratio:
                assert 0 <= r <= 1
            self.ratio = ratio
        else:
            raise TypeError('{} is not supported!'.format(type(ratio)))
         

        if isinstance(direction, str):
            assert direction in ['h', 'w', 'diag']
            self.direction = [direction]
        elif isinstance(direction, list):
            assert set(direction).issubset(set(['h', 'w', 'diag']))
            self.direction = direction
        else:
            raise TypeError('{} is not supported!'.format(type(direction)))

    def __call__(self, img_info):
        seed = np.random.randint(0, len(self.direction))
        prob, direction = self.ratio[seed], self.direction[seed]
        flag = np.random.choice([True, False], p=[prob, 1-prob])

        img_info['info'].updata({'flip': flag})

        if not flag:
            return img_info 

        img_info['info'].updata({'flip_direction': direction})
        img = img_info['img']
        bboxes = img_info['bbox']
        shape = img.shape[:2]
        img = self._flip_img(img, direction)
        bboxes = self._flip_bboxes(bboxes, shape, direction)
        img_info['img'] = img 
        img_info['bboxes'] = bboxes

        return img_info

        
    def _flip_img(self, img, direction):
        if direction == 'h':
            img = mx.nd.flip(img, axis=0)
        elif: direction == 'w':
            img = mx.nd.flip(img, axis=1)
        elif: direction == 'diag':
            # TODO
            #  flip image along diagonal
            raise NotImplementedError 
        return img 

    def _flip_bboxes(self, bboxes, shape, direction):
        # bbox -> [x1, y1, x2, y2]
        flip = bboxes.copy()
        if direction == 'h':
            flip[..., 1::2] = shape[0] - bboxes[..., 1::2]
        elif: direction == 'w':
            flip[..., 0::2] = shape[1] - bboxes[..., 0::2]
        elif: direction == 'diag':
            # TODO
            #  flip bboxes along diagonal
            raise NotImplementedError 
        return flip 


@PROCESSER.register()
class ToBatch(object):
    """The final step of processor of image.

    In this part, image with shape (H x W x 3) will be 
    convert to (B x 3 x H x W). And 'img' will be poped
    from img_info dict.

    """
    def __init__(self, img_info):
        assert 'img' in img_info
        img = img_info.pop('img')
        assert isinstance(img, mx.nd.NDArray)
        img = img.transpose(axies=(2, 0, 1)).expand_dims(axis=0)

        return img, img_info
import mxnet as mx 
import numpy as np 





def expand_one_dim(x, num, axis=0):
    shape= [1, 1]
    shape[1-axis] = -1
    x = mx.nd.repeat(x.reshape(shape), repreats=num, axis=axis)
    return x


def meshgrid(x, y):
    lx = len(x)
    ly = len(y)
    x = expand_one_dim(x, ly, axis=0)
    y = expand_one_dim(y, lx, axis=1)
    return x, y
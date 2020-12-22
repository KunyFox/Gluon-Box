from functools import partial
from six.moves import map, zip

def func_mapping(func, *args, **kwargs):
    """Mapping func to a list of arguments.

    Reference to mmdetection 'multi_apply'.
    
    """
    _func = partial(func, **kwargs) if kwargs else func 
    _results = map(_func, *args)
    _results = tuple(map(list, zip(*_results)))
    return _results
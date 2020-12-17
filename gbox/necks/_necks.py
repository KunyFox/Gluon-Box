import sys 
sys.path.append('.')

from utils import ModuleRegister

NECKS = ModuleRegister('necks')

def get_neck(cfg):
    assert isinstance(cfg, dict), 
        "config must be a dict, but got {}".format(type(cfg))

    assert 'type' in cfg, 
        "cfg should have contribute to 'type'"
    
    _cfg = cfg.copy()
    neck_type = _cfg.pop('type')
    assert isinstance(backbone_type, str),
        "neck_type must be str, but got {}".format(type(neck_type))
    neck = NECKS.get(neck_type)
    return neck(**_cfg)
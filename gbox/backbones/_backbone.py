import sys 
sys.path.append('.')
from utils import ModuleRegister



BACKBONBES = ModuleRegister('backbone')

def get_backbone(cfg):
    assert isinstance(cfg, dict), \
        "config must be a dict, but got {}".format(type(cfg))

    assert 'type' in cfg, \
        "cfg should have contribute to 'type'"
    
    _cfg = cfg.copy()
    backbone_type = _cfg.pop('type')
    assert isinstance(backbone_type, str), \
        "backbone_type must be str, but got {}".format(type(backbone_type))
    backbone = BACKBONBES.get(backbone_type)
    return backbone(**_cfg)

    
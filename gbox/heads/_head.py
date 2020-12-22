import sys 
sys.path.append('.')
from utils import ModuleRegister



HEADS = ModuleRegister('heads')

def get_backbone(cfg):
    assert isinstance(cfg, dict), \
        "config must be a dict, but got {}".format(type(cfg))

    assert 'type' in cfg, \
        "cfg should have contribute to 'type'"
    
    _cfg = cfg.copy()
    head_type = _cfg.pop('type')
    assert isinstance(head_type, str), \
        "backbone_type must be str, but got {}".format(type(head_type))
    backbone = HEADS.get(head_type)
    return backbone(**_cfg)

    
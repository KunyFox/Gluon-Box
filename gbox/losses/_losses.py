import sys 
sys.path.append('.')

from utils import ModuleRegister

LOSSES = ModuleRegister('losses')

def get_neck(cfg):
    assert isinstance(cfg, dict), 
        "config must be a dict, but got {}".format(type(cfg))

    assert 'type' in cfg, 
        "cfg should have contribute to 'type'"
    
    _cfg = cfg.copy()
    loss_type = _cfg.pop('type')
    assert isinstance(loss_type, str),
        "loss_type must be str, but got {}".format(type(loss_type))
    loss = LOSSES.get(loss_type)
    return loss(**_cfg)
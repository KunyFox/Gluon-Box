import sys 
sys.path.append('.')

from utils import ModuleRegister


NECKS = ModuleRegister('necks')
HEADS = ModuleRegister('heads')
LOSSES = ModuleRegister('losses')
BACKBONBES = ModuleRegister('backbones')
DATASETS = ModuleRegister('datasets')
PROCESSERS = ModuleRegister('processers')

def build_from_cfg(cls, cfg):
    assert isinstance(cls, ModuleRegister)
    assert isinstance(cfg, dict), 
        "config must be a dict, but got {}".format(type(cfg))
    assert 'type' in cfg, 
        "cfg should have contribute to 'type'"
    _cfg = cfg.copy()
    _type = _cfg.pop('type')
    assert isinstance(nexk_type, str),
        "neck_type must be str, but got {}".format(type(neck_type))

    instance = cls.get(_type, None)
    assert instance is not None 

    return instance(**_cfg)


def get_backbone(cfg):
    return build_from_cfg(BACKBONBES, cfg)

def get_neck(cfg):
    return build_from_cfg(NECKS, cfg)

def get_head(cfg):
    return build_from_cfg(HEADS, cfg) 

def get_loss(cfg):
    return build_from_cfg(LOSSES, cfg)

def get_dataset(cfg):
    return build_from_cfg(DATASETS, cfg)

def get_processers(cfg):


def get_processers(processers):
    assert isinstance(processers, list), 
        "processer must be a list, but got {}".format(type(processers))
    
    target_pros = [build_from_cfg(PROCESSERS, cfg) for cfg in processers]
    
    assert isinstance(target_pros[-1], PROCESSER.get('ToBatch')),  \
        "Processers should end of 'ToBatch'."

    return target_pros
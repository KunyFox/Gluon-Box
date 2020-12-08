from utils import ModuleRegister

PROCESSER = ModuleRegister('backbone')

def get_processer(processer):
    assert isinstance(cfg, dict), 
        "processer must be a dict, but got {}".format(type(cfg))

    processer_f = []

    for p in processer:
        f = PROCESSER.get(p, None)
        if f is None:
            raise ValueError("{} is not registed in processer!".format(f))
        processer_f.append(f(**processer[p]))
    
    assert isinstance(processer[-1], PROCESSER.get('ToBatch')), "Processer should end of 'ToBatch'."

    return processer_f
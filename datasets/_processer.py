from utils import ModuleRegister

PROCESSER = ModuleRegister('backbone')

def get_processer(processer):
    assert isinstance(cfg, list), 
        "processer must be a list, but got {}".format(type(cfg))

    assert processer[-1] == 'ToBatch', "Processer should end of 'ToBatch'."

    processer_f = []

    for p in processer:
        f = PROCESSER.get(p, None)
        if f is None:
            raise ValueError("{} is not registed in processer!".format(f))
        processer_f.append(f)
    
    return processer_f
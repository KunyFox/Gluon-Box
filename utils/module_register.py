import inspect
import warnings


class ModuleRegister():
    def __init__(self, name):
        self._name = name 
        self._modules = {}

    @property
    def name(self):
        return self._name 

    @property
    def modules(self):
        return list(self._modules.keys())

    def get(self, key):
        assert key in self.modules, "Module {} is not registed".format(key)
        return self._modules.get(key)

    def _register_with_module(self, module, name=None, regis_force=False):
        if not inspect.isclass(module):
            raise TypeError("Module must be a class, but got {}".format(type(module)))

        if name is None:
            name = module.__name__

        if name in self.modules:
            if not regis_force:
                raise KeyError("{} is already registed, if you want to override it, please set regis_force to True".format(name))
            else:
                warnings.warn("Override the module {}".format(name), UserWarning)
        self._modules[name] = module
            

    def register(self, module=None, name=None, regis_force=False):
        assert isinstance(name, str) or name is None, \
            "name must be a str, but got {}".format(type(name))

        if module is not None:
            ## x.register(module=YourClass)
            self._register_with_module(module=module, name=name, regis_force=regis_force)
            return module
        
        ## x.register()
        def _register(module):
            self._register_with_module(module=module, name=name, regis_force=regis_force)
            return module 
        return _register
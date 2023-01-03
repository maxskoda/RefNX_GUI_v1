from copy import deepcopy


class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)


class Parameter(metaclass=IterRegistry):
    _registry = []

    def __init__(self, name, value, minval, maxval):
        self._registry.append(self)
        self.name = name
        self.value = value
        self.minval = minval
        self.maxval = maxval


class Layer(metaclass=IterRegistry):
    _registry = []

    def __init__(self, name, thickness, sld, roughness, hydration):
        self._registry.append(self)
        self.name = name
        self.thickness = thickness
        self.sld = sld
        self.roughness = roughness
        self.hydration = hydration


oxide_thick = Parameter('oxide_thick', 15, 10, 20)
oxide_sld = Parameter('oxide_sld', 3.47, 3.4, 3.5)
oxide_rough = Parameter('oxide_rough', 3.0, 3.0, 10.0)

head_thick = Parameter('head_thick', 8, 5, 20)
head_sld = Parameter('head_sld', 4.5, 3.4, 5.5)


oxide_layer = Layer('oxide_layer', oxide_thick, oxide_sld, oxide_rough, None)
head_layer = Layer('head_layer', head_thick, head_sld, oxide_rough, None)


for l in Layer:
    print(l.name, l.thickness.value, l.sld.value, l.roughness.value)


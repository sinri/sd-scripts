import torch


class TorchRandReplacer:
    def __init__(self, noise_manager):
        self.noise_manager = noise_manager

    def __getattr__(self, item):
        if item == "randn":
            return self.noise_manager.randn
        if hasattr(torch, item):
            return getattr(torch, item)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

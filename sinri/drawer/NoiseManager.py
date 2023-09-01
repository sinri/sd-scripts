import torch

# replace randn
class NoiseManager:
    def __init__(self):
        self.sampler_noises = None
        self.sampler_noise_index = 0

    def reset_sampler_noises(self, noises):
        self.sampler_noise_index = 0
        self.sampler_noises = noises

    def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
        # print("replacing", shape, len(self.sampler_noises), self.sampler_noise_index)
        if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
            noise = self.sampler_noises[self.sampler_noise_index]
            if shape != noise.shape:
                noise = None
        else:
            noise = None

        if noise == None:
            print(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
            noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

        self.sampler_noise_index += 1
        return noise
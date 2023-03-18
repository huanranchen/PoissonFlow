import torch
from torch import nn
from torchvision import transforms
import numpy as np
import math


class VanillaSampler(nn.Module):
    def __init__(self,
                 unet: nn.Module,
                 img_shape=(3, 32, 32),
                 z_max=40,
                 dt=0.1,
                 std=0.5,
                 mean=0.5,
                 gamma=15,
                 ):
        super(VanillaSampler, self).__init__()
        self.device = torch.device('cuda')
        self.unet = unet
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.z_max = z_max
        self.std = std
        self.mean = mean
        self.dt = dt
        self.init()
        self.gamma = gamma

    def init(self):
        self.unet.eval().to(self.device).requires_grad_(False)
        # self.noise_type = "diagonal"
        # self.sde_type = "ito"
        self.to_img = transforms.ToPILImage()
        self.i = 0
        print(f'poisson flow vanilla solver, dt is {self.dt}')

    def initialize(self, batch_size=1):
        gaussian = torch.randn(batch_size, *self.img_shape, device=self.device)
        unit_gaussian = gaussian / torch.norm(gaussian.view(batch_size, -1), p=2, dim=1).view(batch_size, 1, 1, 1)

        # sample angle
        # angle = torch.rand((batch_size,), device=self.device) * math.pi / 2
        # norm_coeff = self.z_max * torch.cos(angle)
        samples_norm = np.random.beta(a=self.state_size / 2. - 0.5, b=0.5, size=batch_size)
        inverse_beta = samples_norm / (1 - samples_norm)
        # Sampling from p_radius(R) by change-of-variable
        samples_norm = np.sqrt(self.z_max ** 2 * inverse_beta)
        # clip the sample norm (radius)
        samples_norm = np.clip(samples_norm, 1, 3000)
        samples_norm = torch.from_numpy(samples_norm).cuda().view(len(samples_norm), -1).to(torch.float)
        print(samples_norm)
        # result
        result = samples_norm.view(batch_size, 1, 1, 1) * unit_gaussian
        return result

    def convert(self, x):
        x = x * self.std + self.mean
        # print(torch.min(x), torch.max(x))
        img = self.to_img(x[0])
        img.save(f'./what/{self.i}.png')
        self.i += 1
        return x

    @torch.no_grad()
    def sample(self):
        x = self.initialize()
        N, C, H, D = x.shape
        assert N == 1, 'sample size now only support 1'
        z = torch.zeros((N,), device=self.device) + self.z_max
        while z[0] > 1e-3:
            normalized_x = self.unet(x, z)
            norm_out = torch.norm(normalized_x)
            # print(direction)
            x = x + normalized_x * self.dt
            normalized_z = z / (torch.sqrt(
                (
                        (self.gamma * norm_out / math.sqrt(self.state_size)) /
                        (1 - norm_out / math.sqrt(self.state_size))
                ) ** 2
                + z ** 2
            )
                                + self.gamma)
            print(z, normalized_z)
            z = z - self.dt * normalized_z * math.sqrt(self.state_size)
        return self.convert(x)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

#

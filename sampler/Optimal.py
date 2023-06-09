import torch
from torch import nn
from torchvision import transforms
import numpy as np
import math


class OptimalSampler():
    def __init__(self,
                 loader,
                 img_shape=(3, 32, 32),
                 z_max=40,
                 stride=0.05,
                 std=0.5,
                 mean=0.5,
                 ):
        super(OptimalSampler, self).__init__()
        self.device = torch.device('cuda')
        self.loader = loader
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.z_max = z_max
        self.std = std
        self.mean = mean
        self.stride = stride
        self.init()

    def init(self):
        # self.noise_type = "diagonal"
        # self.sde_type = "ito"
        self.to_img = transforms.ToPILImage()
        self.i = 0
        print(f'poisson flow vanilla solver, dt is {self.stride}')
        # prepare xs
        xs = []
        for x, _ in self.loader:
            xs.append((x.cuda() - 0.5) * 2)
        self.xs = torch.cat(xs, dim=0)

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
        # print(samples_norm)
        # result
        result = samples_norm.view(batch_size, 1, 1, 1) * unit_gaussian
        # print(result)
        return unit_gaussian * 1000

    def convert(self, x):
        x = x * self.std + self.mean
        # print(torch.min(x), torch.max(x))
        img = self.to_img(x[0])
        img.save(f'./what/{self.i}.png')
        self.i += 1
        return x

    def get_optimal_v(self, perturbed_x):
        batch_x = self.xs
        BL, C, H, D = batch_x.shape
        N = perturbed_x.shape[0]
        real_difference = - (perturbed_x.unsqueeze(1) - batch_x)  # N, BL, C, H, D
        # calculate weight for each sample
        distances = torch.sum((real_difference ** 2).view(N, BL, C * H * D), dim=-1).sqrt()  # N, BL
        # For numerical stability, timing each row by its minimum value
        coeff = torch.min(distances, dim=1, keepdim=True)[0] / (distances + 1e-7)  # N, BL
        coeff = coeff ** (C * H * D)
        coeff = coeff / (torch.sum(coeff, dim=1, keepdim=True) + 1e-7)  # N, BL

        # ground truth
        # print(coeff.shape, real_difference.shape)
        gt = coeff.view(N, BL, 1, 1, 1) * real_difference  # N, BL, C, H, D
        gt = torch.sum(gt, dim=1)  # N, C, H, D
        gt_norm = torch.norm(gt.view(N, C * H * D), dim=1, p=2).view(N, 1, 1, 1)
        gt = gt / (gt_norm + 5)
        gt = gt * math.sqrt(C * H * D)
        return gt

    @torch.no_grad()
    def sample(self, ):
        x = self.initialize()
        N, C, H, D = x.shape
        z = torch.zeros((N,), device=self.device) + self.z_max
        while z[0] > 0:
            v = self.get_optimal_v(x)
            # print(direction)
            x = x + v * self.stride
            z = z - self.stride
        return self.convert(x)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

#

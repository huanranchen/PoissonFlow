import torch
import torchsde
from torch import nn
from torchvision import transforms
from torch import Tensor
from copy import deepcopy


class DiffusionSde(nn.Module):
    def __init__(self, unet: nn.Module, beta=None, img_shape=None, T=1000, dt=1e-3,
                 w=3, mode='imagenet'):
        super(DiffusionSde, self).__init__()
        if beta is None:
            beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda'))
        if img_shape is None:
            img_shape = (3, 256, 256)
        self.device = torch.device('cuda')
        self.unet = unet
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.beta = beta * T
        self.T = T
        self.dt = dt
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.w = w
        self.init()
        self.mode = mode

    def init(self):
        self.eval().to(self.device).requires_grad_(False)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.to_img = transforms.ToPILImage()
        self.i = 0
        print(f'dt is {self.dt}')

    def convert(self, x):
        if self.mode == 'imagenet':
            x = x * torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1) + \
                torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        else:
            x = (x + 1) * 0.5
        img = self.to_img(x[-1][0] if len(x.shape) == 5 else x[0])
        img.save(f'./what/{self.i}.png')
        self.i += 1
        return x

    def diffusion_forward(self, x: torch.Tensor, t: int):
        assert len(x.shape) == 2, 'x should be N, D'
        N = x.shape[0]
        diffusion = torch.sqrt(self.beta[t]).view(1, 1).repeat(N, x.shape[1])
        drift = -0.5 * self.beta[t] * x
        return drift, diffusion

    def reverse_diffusion_forward(self, x: torch.int, t: int, return_type='diffusion'):
        N = x.shape[0]
        tensor_t = torch.zeros((N,), device=self.device, dtype=torch.int) + (self.T - t - 1)
        forward_drift, forward_diffusion = self.diffusion_forward(x, self.T - t - 1)
        diffusion = forward_diffusion
        if return_type == 'diffusion':
            return diffusion
        if 'y' in self.diffusion_kwargs:
            pre_with_y = self.unet(x.view(N, *self.img_shape), tensor_t,
                                   **self.diffusion_kwargs)[:, :3, :, :].view(N, -1)
            kwargs_without_y = deepcopy(self.diffusion_kwargs)
            kwargs_without_y.pop('y')
            pre_without_y = self.unet(x.view(N, *self.img_shape), tensor_t,
                                      **kwargs_without_y)[:, :3, :, :].view(N, -1)
            pre = pre_with_y * (self.w + 1) - pre_without_y * self.w
        else:
            pre = self.unet(x.view(N, *self.img_shape), tensor_t,
                            **self.diffusion_kwargs)[:, :3, :, :].view(N, -1)
        score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])
        drift = forward_drift - diffusion ** 2 * score
        return -drift

    def f(self, t: float, x):
        f = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='drift')
        assert f.shape == x.shape
        return f

    def g(self, t: float, x):
        g = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='diffusion')
        return g

    @torch.no_grad()
    def sample(self, y):
        self.y = y
        x = torch.randn((1, *self.img_shape), device=self.device).view((1, self.state_size))
        ts = torch.tensor([0., 1. - 1e-4], device=self.device)
        x = torchsde.sdeint(self, x, ts, method='euler')
        x = x[-1]  # N, 3, 256, 256
        x = x.reshape(1, *self.img_shape)
        self.y = None
        return self.convert(x)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


class DiffusionOde(DiffusionSde):
    def __init__(self, *args, **kwargs):
        super(DiffusionOde, self).__init__(*args, **kwargs)

    def reverse_diffusion_forward(self, x: torch.int, t: int, return_type='diffusion'):
        N = x.shape[0]
        tensor_t = torch.zeros((N,), device=self.device, dtype=torch.int) + (self.T - t - 1)
        forward_drift, forward_diffusion = self.diffusion_forward(x, self.T - t - 1)
        diffusion = forward_diffusion
        if return_type == 'diffusion':
            return diffusion
        pre = self.unet(x.view(N, *self.img_shape), tensor_t)[:, :3, :, :].view(N, -1)
        score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])
        drift = forward_drift - 0.5 * diffusion ** 2 * score
        return -drift

    def f(self, t: torch.tensor, x):
        f = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='drift')
        assert f.shape == x.shape
        # print('f', torch.sum(f))
        return f

    def g(self, t: float, x):
        return torch.zeros_like(x)

import torch
import torchsde
from torch import nn
from torchvision import transforms
from torch import Tensor
from copy import deepcopy
from torch.utils.data import DataLoader


class DiffusionSdeOptimal(nn.Module):
    def __init__(self,
                 loader: DataLoader,
                 beta=None,
                 img_shape=None,
                 T=1000,
                 dt=1e-3,
                 mode='cifar',
                 condition=True,
                 num_classes=10, ):
        super(DiffusionSdeOptimal, self).__init__()
        self.loader = loader
        if beta is None:
            beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda'))
        if img_shape is None:
            img_shape = (3, 32, 32)
        self.device = torch.device('cuda')
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.beta = beta * T
        self.T = T
        self.dt = dt
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.init()
        self.mode = mode
        self.condition = condition
        self.num_classes = num_classes

    def init(self):
        self.eval().to(self.device).requires_grad_(False)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.to_img = transforms.ToPILImage()
        self.i = 0
        print(f'dt is {self.dt}')
        self.x = []
        self.y = []
        for now_x, now_y in self.loader:
            self.x.append(now_x.cuda())
            self.y.append(now_y.cuda())
        self.x = torch.cat(self.x, dim=0)
        self.x = (self.x - 0.5) * 2
        self.y = torch.cat(self.y, dim=0)

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

    def reverse_diffusion_forward(self, x: Tensor, t: int, return_type='diffusion'):
        N = x.shape[0]
        tensor_t = torch.zeros((N,), device=self.device, dtype=torch.long) + (self.T - t - 1)
        forward_drift, forward_diffusion = self.diffusion_forward(x, self.T - t - 1)
        diffusion = forward_diffusion
        if return_type == 'diffusion':
            return diffusion
        # score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])\
        score = self.get_optimal_score_with_t(x.view(N, *self.img_shape), tensor_t).view(N, -1)
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
    def sample(self):
        self.step_count = 0
        self.diffusion_kwargs = {}
        x = (torch.randn((1, *self.img_shape), device=self.device).view((1, self.state_size)))
        # x = (x + torch.min(x)) / torch.max(x)
        # x = (x + 1) / 2
        ts = torch.tensor([0., 1. - 1e-4], device=self.device)
        x = torchsde.sdeint(self, x, ts, method='euler')
        x = x[-1]  # N, 3, 256, 256
        x = x.reshape(1, *self.img_shape)
        return self.convert(x)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    # def get_optimal_unet_predict(self, u: Tensor):
    #     """
    #
    #     :param u: 1, C, H, D
    #     :return:
    #     """
    #     criterion = nn.MSELoss(reduction='none')
    #     p = torch.randn_like(u, requires_grad=True)
    #     optimizer = torch.optim.Adam([p], lr=1e-4)
    #     loader = self.loader
    #     for x, y in loader:
    #         x = x.cuda().unsqueeze(1).repeat(1, 1000, 1, 1, 1)  # N, T, C, H, D
    #         N, T, C, H, D = x.shape
    #         t = torch.arange(1000, device=self.device)
    #         target = (u - torch.sqrt(self.alpha_bar[t]).view(1, 1000, 1, 1, 1) * x) / torch.sqrt(
    #             1 - torch.sqrt(self.alpha_bar[t]).view(1, 1000, 1, 1, 1))  # N, T, C, H, D
    #         prob = self.standard_gaussian_pdf(target.view(N * T, C, H, D))  # N*T
    #         target = target.view(N * T, C, H, D)
    #         loss = criterion(p.squeeze(), target).mean(1).mean(1).mean(1)  # N*T
    #         loss = prob * loss
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     return p.detach()

    def get_optimal_score_with_t(self, u: Tensor, t: Tensor):
        """

        :param u: 1, C, H, D
        :param t:1
        :return: if not condition, return same shape like u.
                if condition, return num_classes, *u.shape
        """
        B = u.shape[0]
        if not self.condition:
            N, C, H, D = self.x.shape
            sigma_t = torch.sqrt(1 - self.alpha_bar[t])
            inner_exp = - 1 / (2 * sigma_t ** 2) * \
                        torch.sum((
                                          (u.squeeze() -
                                           torch.sqrt(self.alpha_bar[t].view(1, 1, 1, 1)) * self.x) ** 2).view(N,
                                                                                                               C * H * D),
                                  dim=1).squeeze()  # N
            softmaxs = torch.softmax(inner_exp, dim=0)  # N
            outer_weight = -1 / sigma_t ** 2 * (u.squeeze() - torch.sqrt(self.alpha_bar[t]).view(1, 1, 1, 1) * self.x)
            outer_weight = outer_weight.permute(1, 2, 3, 0)  # C, H, D, N
            result = outer_weight @ softmaxs
            result = result.unsqueeze(0)
            return result
        else:
            result = []
            for now_class in range(self.num_classes):
                mask = self.y == now_class
                x = self.x[mask]
                N, C, H, D = x.shape
                sigma_t = torch.sqrt(1 - self.alpha_bar[t])
                inner_exp = - 1 / (2 * sigma_t ** 2) * \
                            torch.sum((
                                              (u.squeeze() -
                                               torch.sqrt(self.alpha_bar[t].view(1, 1, 1, 1)) * x) ** 2).view(N,
                                                                                                              C * H * D),
                                      dim=1).squeeze()  # N
                softmaxs = torch.softmax(inner_exp, dim=0)  # N
                outer_weight = -1 / sigma_t ** 2 * (
                        u.squeeze() - torch.sqrt(self.alpha_bar[t]).view(1, 1, 1, 1) * x)
                outer_weight = outer_weight.permute(1, 2, 3, 0)  # C, H, D, N
                this_class_result = outer_weight @ softmaxs
                this_class_result = this_class_result.unsqueeze(0)
                result.append(this_class_result)
                # mask = self.y == now_class
                # x = self.x[mask]
                # N, C, H, D = x.shape
                # sigma_t = torch.sqrt(1 - self.alpha_bar[t])
                # inner_exp = - 1 / (2 * sigma_t ** 2) * \
                #             torch.sum((
                #                               (u.unsqueeze(1).repeat(1, N, 1, 1, 1) -  # B, N, C, H, D
                #                                torch.sqrt(self.alpha_bar[t].view(1, 1, 1, 1)) * x) ** 2).view(B, N,
                #                                                                                               C * H * D),
                #                       dim=2).squeeze()  # B, N
                # softmaxs = torch.softmax(inner_exp, dim=1)  # B, N
                # outer_weight = -1 / sigma_t ** 2 * (
                #         u.unsqueeze(1).repeat(1, N, 1, 1, 1) -
                #         torch.sqrt(self.alpha_bar[t]).view(1, 1, 1, 1) * x)  # B, N, C, H, D
                # outer_weight = outer_weight.permute(0, 2, 3, 4, 1)  # B, C, H, D, N
                # this_class_result = torch.bmm(outer_weight.view(B, C * H * D, N),  # (B, C, H, D, N) x (B, N, 1)
                #                               softmaxs.unsqueeze(2)).view(B, C, H, D)  # B, C, H, D
                # this_class_result = this_class_result.unsqueeze(0)
                # result.append(this_class_result)
            result = torch.stack(result)
            return result

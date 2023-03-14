import torch
from torch import nn
import random
from tqdm import tqdm
import math


class PoissonFlowSolver():
    def __init__(self, unet: nn.Module,
                 criterion=nn.MSELoss(),
                 device=torch.device('cuda'),
                 small_batch_size=64):
        self.unet = unet.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=2e-4)
        self.small_batch_size = small_batch_size

    def transform(self, x):
        return (x - 0.5) * 2

    def init(self):
        # init
        self.unet.eval().requires_grad_(False).to(self.device)

    def train(self, train_loader, total_epoch=1000, p_uncondition=1,
              fp16=False):
        self.unet.train()
        self.unet.requires_grad_(True)
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        for epoch in range(1, total_epoch + 1):
            epoch_loss = 0
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.cuda(), y.cuda()
                # some preprocess
                x = self.transform(x)
                # train
                x, y = x.to(self.device), y.to(self.device)

                for i in range(math.ceil(x.shape[0] / self.small_batch_size)):
                    if fp16:
                        with autocast():
                            loss = self.batch_fn(x[i * self.small_batch_size:(i + 1) * self.small_batch_size], x)
                    else:
                        loss = self.batch_fn(x[i * self.small_batch_size:(i + 1) * self.small_batch_size], x)
                    if fp16:
                        raise NotImplementedError
                        pass
                    else:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'step {step}, loss {epoch_loss / step}')
            print(f'epoch {epoch}, loss {epoch_loss / len(train_loader)}')
            torch.save(self.unet.state_dict(), 'unet.pt')

        self.init()

    def batch_fn(self, x, batch_x, small_batch_size=64):
        BL, C, H, D = batch_x.shape
        N = small_batch_size
        perturbed_x, perturbed_z = self.perturb(x)
        real_difference = - (perturbed_x.unsqueeze(1) - batch_x)  # N, BL, C, H, D
        # calculate weight for each sample
        distances = torch.sum((real_difference ** 2).view(N, BL, C * H * D), dim=-1)  # N, BL
        # print(distances.shape, )
        coeff = torch.min(distances, dim=1, keepdim=True)[0] / (
                torch.sum(distances, dim=1, keepdim=True) + 1e-7)  # N, 1
        # ground truth
        # print(coeff.shape, real_difference.shape)
        gt = coeff.view(N, 1, 1, 1, 1) * real_difference  # N, BL, C, H, D
        gt = torch.sum(gt, dim=1)  # N, C, H, D
        gt_norm = torch.norm(gt.view(N, C * H * D), dim=1).view(N, 1, 1, 1)
        gt = gt / gt_norm
        gt = gt * math.sqrt(C * H * D)
        # final loss
        predict = self.unet(x, perturbed_z)[:, :3, :, :]
        loss = self.criterion(predict, gt)
        return loss

    def perturb(self, x, M=291, sigma=0.01, tao=0.03):
        N, C, H, D = x.shape
        m = random.random() * M
        multiplier = (1 + tao) ** m
        # eps z
        eps_z = torch.abs(torch.randn((N,), device=self.device) * sigma) * multiplier
        # eps x
        eps_x = torch.norm((torch.randn_like(x, device=self.device) *
                            sigma).view(N, C * H * D), dim=1, keepdim=True, p=2).view(N, 1, 1, 1)
        unit_gaussian = torch.randn_like(x)
        unit_gaussian = unit_gaussian / torch.norm(unit_gaussian.view(N, C * H * D), dim=1, p=2).view(N, 1, 1, 1)
        eps_x = eps_x * unit_gaussian * multiplier
        return x + eps_x, eps_z

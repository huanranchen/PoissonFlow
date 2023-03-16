import torch
from sampler import VanillaSampler, OptimalSampler
from model import get_NCSNPP
from solver import PoissonFlowSolver
from data import get_CIFAR10_train, get_CIFAR10_test
from tester import most_similar
from torchvision import transforms

to_img = transforms.ToPILImage()



train_loader = get_CIFAR10_train(batch_size=2048)
model = get_NCSNPP()
model.load_state_dict(torch.load('unet.pt'))
solver = PoissonFlowSolver(model)
solver.train(train_loader)

# #
sampler = VanillaSampler(unet=model)
sampler.sample()
# sampler = OptimalSampler(train_loader)
# x = sampler.sample()
# target = most_similar(x, train_loader)
# img = to_img(target.squeeze())
# img.save('./what/0_target.png')
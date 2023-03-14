import torch

from model import get_NCSNPP
from solver import PoissonFlowSolver
from data import get_CIFAR10_train


train_loader = get_CIFAR10_train(batch_size=1024)
model = get_NCSNPP()
model.load_state_dict(torch.load('unet.pt'))
solver = PoissonFlowSolver(model)
solver.train(train_loader)

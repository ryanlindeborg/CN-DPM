from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter
from torch import nn as nn


# ==========
# Model ABCs
# ==========

class Model(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def learn(self, x, y, t, step):
        pass

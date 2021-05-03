import yaml
import torch
from torch import nn
from ndpm import Ndpm
from .base import Model


class NdpmModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.ndpm = Ndpm(config)
        self.extractor = None

    def forward(self, x, expert_index=None, return_assignments=False):
        x = x.to(self.device)
        return (
            self.ndpm(x, return_assignments) if expert_index is None else
            self.ndpm.experts[expert_index](x)
        )

    def learn(self, x, y, t, step=None):
        x, y = x.to(self.device), y.to(self.device)
        self.ndpm.learn(x, y, step)

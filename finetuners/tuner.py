from abc import abstractmethod, ABC
from torch import nn
from typing import Tuple

import torch

from train import train_on_batch
import config

from runspec import Sim2RealSpec


class Tuner(ABC):
    def __init__(self, initial_model, objective, opt, lr: float):
        self.model = self.modify_model(initial_model)
        self.objective = objective
        self.opt = opt(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, "max", factor=0.3
        )

    @abstractmethod
    def modify_model(self, initial_model) -> nn.Module:
        return

    def train_on_batch(self, batch, state) -> Tuple[any, float, float]:
        return train_on_batch(state, self.model, self.opt, self.objective, batch)

    def scheduler_step(self, val_loss):
        self.scheduler.step(val_loss)

    @abstractmethod
    def name(self) -> str:
        pass

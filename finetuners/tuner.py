from abc import abstractmethod, ABC
from torch import nn
from typing import Tuple

from train import train_on_batch

from runspec import Sim2RealSpec


class Tuner(ABC):
    def __init__(self, initial_model, objective, opt, spec: Sim2RealSpec):
        self.model = self.modify_model(initial_model)
        self.objective = objective
        self.opt = opt(self.model.parameters(), spec.opt.lr)

    @abstractmethod
    def modify_model(self, initial_model) -> nn.Module:
        return

    def train_on_batch(self, batch, state) -> Tuple[any, float, float]:
        return train_on_batch(state, self.model, self.opt, self.objective, batch)

    @abstractmethod
    def name(self) -> str:
        pass

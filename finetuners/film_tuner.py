from finetuners.tuner import Tuner
from torch import nn
from typing import Tuple
import lab as B

from components.film import FiLM


class FilmTuner(Tuner):
    def __init__(self, initial_model, objective, opt, spec):
        super().__init__(initial_model, objective, opt, spec)

    def modify_model(self, initial_model: nn.Module) -> nn.Module:
        initial_model.requires_grad_(False)
        for i, module in enumerate(initial_model.modules()):
            if type(module) == FiLM:
                module.requires_grad_(True)
        return initial_model

    def name(self) -> str:
        return "FilmTuner"

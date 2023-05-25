from finetuners.tuner import Tuner
from torch import nn
from typing import Tuple
import lab as B

from components.film import FiLM


class FilmLinearTuner(Tuner):
    def __init__(self, initial_model, objective, opt, spec):
        lr = spec.opt.lr * 10
        self.spec = spec
        super().__init__(initial_model, objective, opt, lr)

    def modify_model(self, initial_model: nn.Module) -> nn.Module:
        initial_model.requires_grad_(False)
        for i, module in enumerate(initial_model.modules()):
            if type(module) == FiLM:
                module.requires_grad_(True)

        if self.spec.model.arch == "unet":
            unet = initial_model.decoder[0]
            unet.final_linear.requires_grad_(True)
        else:
            raise NotImplementedError(
                f"Arch {self.spec.model.arch} is not supported with FilmLinearTuner."
            )

        return initial_model

    def name(self) -> str:
        return "FilmLinearTuner"

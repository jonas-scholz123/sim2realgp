from finetuners.tuner import Tuner
from torch import nn
from typing import Tuple
import lab as B

from components.convnet import FiLM


class FilmTuner(Tuner):
    def __init__(self, initial_model, objective, opt, lr):
        super().__init__(initial_model, objective, opt, lr)

    def modify_model(self, initial_model: nn.Module) -> nn.Module:
        initial_model.requires_grad_(False)
        for i, module in enumerate(initial_model.modules()):
            if type(module) == FiLM:
                module.requires_grad_(True)
        return initial_model

    def train_on_batch(self, batch, state) -> Tuple[any, float]:
        state, obj = self.objective(
            state,
            self.model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
        )
        val = -B.mean(obj)
        val.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return state, val

    def name(self) -> str:
        return "FilmTuner"

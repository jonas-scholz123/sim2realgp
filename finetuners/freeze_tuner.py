from finetuners.tuner import Tuner
from torch import nn
from lab import B

from typing import Tuple


class FreezeTuner(Tuner):
    def __init__(self, initial_model, objective, opt, lr, num_layers_tuned=2):
        self.num_layers_tuned = num_layers_tuned
        super().__init__(initial_model, objective, opt, lr)

    def modify_model(self, initial_model: nn.Module) -> nn.Module:
        for layer in initial_model.decoder[0].net[-self.num_layers_tuned :]:
            layer.requires_grad_(False)
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
        return "FreezeTuner"

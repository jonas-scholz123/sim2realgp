from finetuners.tuner import Tuner
from torch import nn
from lab import B
from train import train_on_batch

from typing import Tuple
from runspec import Sim2RealSpec


class FreezeTuner(Tuner):
    def __init__(
        self, initial_model, objective, opt, s: Sim2RealSpec, num_layers_tuned=2
    ):
        self.num_layers_tuned = num_layers_tuned
        self.spec = s
        super().__init__(initial_model, objective, opt, s.opt.lr)

    def modify_model(self, initial_model: nn.Module) -> nn.Module:
        initial_model.requires_grad_(False)

        if self.spec.model.arch == "conv":
            for layer in initial_model.decoder[0].net[-self.num_layers_tuned :]:
                layer.requires_grad_(True)

        elif self.spec.model.arch == "unet":
            unet = initial_model.decoder[0]
            for layer in unet.after_turn_layers[-self.num_layers_tuned :]:
                layer.requires_grad_(True)

            unet.final_linear.requires_grad_(True)

        return initial_model

    def name(self) -> str:
        return "FreezeTuner"

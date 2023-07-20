"""
Utility functions used for training
(both sim and sim2real).
"""
import neuralprocesses.torch as nps
import torch
import lab as B
import numpy as np

from typing import Tuple
from runspec import DataSpec


def train(state, model, opt, objective, gen):
    """Train for an epoch."""
    vals = []
    gp_vals = []
    for batch in gen.epoch():
        state, val, gp_val = train_on_batch(state, model, opt, objective, batch)
        vals.append(val)
        gp_vals.append(gp_val)

    return state, np.mean(vals), np.mean(gp_vals)


def train_on_batch(state, model, opt, objective, batch) -> Tuple[any, float]:
    state, obj = objective(
        state,
        model,
        batch["contexts"],
        batch["xt"],
        batch["yt"],
    )
    val = -B.mean(obj)
    val.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    if "pred_logpdf_diag" in batch:
        n = nps.num_data(batch["xt"], batch["yt"])
        gp_val = B.mean(batch["pred_logpdf_diag"] / n)
    else:
        gp_val = 0.0

    return state, float(-val), float(gp_val)


def evaluate(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():
        vals, gp_vals = [], []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
            )

            # Save numbers.
            vals.append(B.to_numpy(obj))
            if "pred_logpdf_diag" in batch:
                n = nps.num_data(batch["xt"], batch["yt"])
                gp_vals.append(B.to_numpy(batch["pred_logpdf_diag"] / n))

        # Report numbers.
        vals = B.concat(*vals)
        gp_vals = B.concat(*gp_vals)
        # return (
        #    state,
        #    B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals)),
        #    B.mean(gp_vals),
        # )
        return state, B.mean(vals), B.mean(gp_vals)


class MultilengthGPGenerator(nps.GPGenerator):
    """
    Wraps nps.GPGenerator to sample multiple lengthscales.
    """

    def __init__(self, s: DataSpec, num_tasks, device, seed):
        self.s = s

        if isinstance(s.lengthscale, float):
            self.static_lengthscale = True
            lengthscale = s.lengthscale

        elif isinstance(s.lengthscale, tuple):
            self.base_kernel = s.kernel
            self.static_lengthscale = False
            self.lengthscale_dist = nps.UniformContinuous(*s.lengthscale)
            lengthscale = 1.0

        kernel = s.kernel().stretch(lengthscale)

        super().__init__(
            torch.float32,
            seed=seed,
            noise=s.noise,
            kernel=kernel,
            num_context=self._num_context(lengthscale),
            num_target=self._num_target(lengthscale),
            num_tasks=num_tasks,
            pred_logpdf=True,
            pred_logpdf_diag=True,
            device=device,
        )

    def _num_context(self, lengthscale: float):
        num_context_min = int((1 / lengthscale) * self.s.dim_x)
        num_context_max = int((10 / lengthscale) * self.s.dim_x)
        return nps.UniformDiscrete(num_context_min, num_context_max)

    def _num_target(self, lengthscale: float):
        num_target = int((15 / lengthscale) * self.s.dim_x)
        return nps.UniformDiscrete(num_target, num_target)

    def generate_batch(self):
        if not self.static_lengthscale:
            # Sample a lengthscale
            self.state, lengthscale = self.lengthscale_dist.sample(
                self.state, self.float64
            )

            # Adjust GPGenerator
            self.num_context = self._num_context(lengthscale)
            self.num_target = self._num_target(lengthscale)
            self.kernel = self.base_kernel().stretch(lengthscale)

        return super().generate_batch()


def setup(s: DataSpec, device):
    # Architecture choices specific for the GP experiments:
    # Other settings specific to the GP experiments:
    gen_train = MultilengthGPGenerator(s, s.num_tasks_train, device, s.train_seed)
    gen_cv = lambda: MultilengthGPGenerator(
        s, s.num_tasks_val, device, s.train_seed + 10
    )
    gen_eval = lambda: MultilengthGPGenerator(s, s.num_tasks_val, device, 30)
    return gen_train, gen_cv, gen_eval


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, epochs_passed=1):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += epochs_passed
            if self.counter >= self.patience:
                return True
        return False

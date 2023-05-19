"""
Utility functions used for training
(both sim and sim2real).
"""
import neuralprocesses.torch as nps
import torch
import lab as B
import numpy as np

from typing import Tuple


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

    if "pred_logpdf" in batch:
        n = nps.num_data(batch["xt"], batch["yt"])
        gp_val = B.mean(batch["pred_logpdf"] / n)
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
            if "pred_logpdf" in batch:
                n = nps.num_data(batch["xt"], batch["yt"])
                gp_vals.append(B.to_numpy(batch["pred_logpdf"] / n))

        # Report numbers.
        vals = B.concat(*vals)
        gp_vals = B.concat(*gp_vals)
        return (
            state,
            B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals)),
            B.mean(gp_vals),
        )
        # return state, B.mean(vals), B.mean(gp_vals)


def setup(config, *, num_tasks_train, num_tasks_val, lengthscale):
    # Architecture choices specific for the GP experiments:
    # Other settings specific to the GP experiments:
    kernel = config["kernel"].stretch(lengthscale)

    dim_x = config["dim_x"]
    num_context_min = int((1 / lengthscale) * dim_x)
    num_context_max = int((10 / lengthscale) * dim_x)
    num_target = int((15 / lengthscale) * dim_x)

    gen_train = nps.GPGenerator(
        torch.float32,
        seed=10,
        noise=config["noise"],
        kernel=kernel,
        num_context=nps.UniformDiscrete(num_context_min, num_context_max),
        num_target=nps.UniformDiscrete(num_target, num_target),
        num_tasks=num_tasks_train,
        pred_logpdf=True,
        pred_logpdf_diag=False,
        device=config["device"],
    )

    gen_cv = lambda: nps.GPGenerator(
        torch.float32,
        seed=20,
        noise=config["noise"],
        kernel=kernel,
        num_context=nps.UniformDiscrete(num_context_min, num_context_max),
        num_target=nps.UniformDiscrete(num_target, num_target),
        num_tasks=num_tasks_val,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        device=config["device"],
    )

    gen_eval = lambda: nps.GPGenerator(
        torch.float32,
        seed=30,
        noise=config["noise"],
        kernel=kernel,
        num_context=nps.UniformDiscrete(num_context_min, num_context_max),
        num_target=nps.UniformDiscrete(num_target, num_target),
        num_tasks=num_tasks_val,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        device=config["device"],
    )
    return gen_train, gen_cv, gen_eval

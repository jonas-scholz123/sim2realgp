"""
Utility functions used for training
(both sim and sim2real).
"""
import neuralprocesses.torch as nps
import torch
import lab as B
import wandb

from utils import with_err


def train(state, model, opt, objective, gen, *, fix_noise):
    """Train for an epoch."""
    vals = []
    for batch in gen.epoch():
        state, obj = objective(
            state,
            model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
            fix_noise=fix_noise,
        )
        vals.append(B.to_numpy(obj))
        # Be sure to negate the output of `objective`.
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()

    vals = B.concat(*vals)
    print("Loglik (T)", with_err(vals, and_lower=True))
    return state, B.mean(vals)


def evaluate(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():
        vals, kls, kls_diag = [], [], []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
            )

            # Save numbers.
            n = nps.num_data(batch["xt"], batch["yt"])
            vals.append(B.to_numpy(obj))
            if "pred_logpdf" in batch:
                kls.append(B.to_numpy(batch["pred_logpdf"] / n - obj))
            if "pred_logpdf_diag" in batch:
                kls_diag.append(B.to_numpy(batch["pred_logpdf_diag"] / n - obj))

        # Report numbers.
        vals = B.concat(*vals)
        print("Loglik (V)", with_err(vals, and_lower=True))
        if kls:
            print("KL (full)", with_err(B.concat(*kls), and_upper=True))
        if kls_diag:
            print("KL (diag)", with_err(B.concat(*kls_diag), and_upper=True))

        return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))


def setup(config, *, num_tasks_train, num_tasks_val, lengthscale):
    # Architecture choices specific for the GP experiments:
    # Other settings specific to the GP experiments:
    kernel = config["kernel"].stretch(lengthscale)

    num_context_min = int((1 / lengthscale) * dim_x)
    num_context_max = int((10 / lengthscale) * dim_x)
    num_target = int((15 / lengthscale) * dim_x)

    dim_x = config["dim_x"]
    gen_train = nps.GPGenerator(
        torch.float32,
        seed=10,
        noise=config["noise"],
        kernel=kernel,
        num_context=nps.UniformDiscrete(num_context_min, num_context_max),
        num_target=nps.UniformDiscrete(num_target, num_target),
        num_tasks=num_tasks_train,
        pred_logpdf=False,
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

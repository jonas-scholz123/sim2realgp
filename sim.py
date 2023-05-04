#%%
from functools import partial
import warnings

import neuralprocesses.torch as nps
import numpy as np
import torch
import lab as B
import stheno
from matrix.util import ToDenseWarning

from utils import ensure_exists, with_err
from plot import visualise
from config import config
#%%
warnings.filterwarnings("ignore", category=ToDenseWarning)

train_plot_dir = config["train_plot_dir"]
sim_model_dir = config["sim_model_dir"]
exp_dir = config["exp_dir"]

best_model_path = config["sim_best_model_path"]
latest_model_path = config["sim_latest_model_path"]

ensure_exists(train_plot_dir)
ensure_exists(sim_model_dir)
print("Working dir: ", exp_dir)

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
    return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))

def eval(state, model, objective, gen):
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

def setup(config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    # Architecture choices specific for the GP experiments:
    # TODO: We should use a stride of 1 in the first layer, but for compatibility
    #    reasons with the models we already trained, we keep it like this.
    config["unet_strides"] = (2,) * 6
    config["conv_receptive_field"] = 4
    config["margin"] = 0.1
    dim_x = config["dim_x"]
    if dim_x == 1:
        config["points_per_unit"] = 64
    elif dim_x == 2:
        # Reduce the PPU to reduce memory consumption.
        config["points_per_unit"] = 32
        # Since the PPU is reduced, we can also take off a layer of the UNet.
        config["unet_strides"] = config["unet_strides"][:-1]
        config["unet_channels"] = config["unet_channels"][:-1]
    else:
        raise RuntimeError(f"Invalid input dimensionality {dim_x}.")

    # Other settings specific to the GP experiments:
    config["transform"] = None
    config["plot"] = {
        1: {"range": (-2, 4), "axvline": [2]},
        2: {"range": ((-2, 2), (-2, 2))},
    }

    kernel = config["kernel"].stretch(config["lengthscale_sim"])

    gen_train = nps.GPGenerator(
        torch.float32,
        seed=10,
        noise=config["noise"],
        kernel=kernel,
        num_context=nps.UniformDiscrete(3, 30 * dim_x),
        num_target=nps.UniformDiscrete(50 * dim_x, 50 * dim_x),
        num_tasks=num_tasks_train,
        pred_logpdf=False,
        pred_logpdf_diag=False,
        device=device,
    )

    gen_cv = lambda: nps.GPGenerator(
        torch.float32,
        seed=20,
        noise=config["noise"],
        kernel=kernel,
        num_context=nps.UniformDiscrete(3, 30 * dim_x),
        num_target=nps.UniformDiscrete(50 * dim_x, 50 * dim_x),
        num_tasks=num_tasks_cv,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        device=device,
    )

    gen_eval = lambda: nps.GPGenerator(
        torch.float32,
        seed=30,
        noise=config["noise"],
        kernel=kernel,
        num_context=nps.UniformDiscrete(3, 30 * dim_x),
        num_target=nps.UniformDiscrete(50 * dim_x, 50 * dim_x),
        num_tasks=num_tasks_eval,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        device=device,
    )
    return gen_train, gen_cv, gen_eval

device = config["device"]
B.set_global_device(device)

gen_train, gen_cv, gens_eval = setup(
    config,
    num_tasks_train=config["num_tasks_train"],
    num_tasks_cv=config["num_tasks_val"],
    num_tasks_eval=config["num_tasks_val"],
    device=device,
)

model = nps.construct_gnp(
    dim_x=config["dim_x"],
    dim_yc=(1,) * config["dim_y"],
    dim_yt=config["dim_y"],
    dim_embedding=config["dim_embedding"],
    enc_same=config["enc_same"],
    num_dec_layers=config["num_layers"],
    width=config["width"],
    likelihood="lowrank",
    num_basis_functions=config["num_basis_functions"],
    transform=config["transform"],
)

objective = partial(
    nps.loglik,
    num_samples=config["num_samples"],
    normalise=config["normalise_obj"],
)

model = model.to(device)
opt = torch.optim.Adam(model.parameters(), config["rate"])

state = B.create_random_state(torch.float32, seed=0)
fix_noise=None

best_eval_lik = -np.infty

for i in range(config["num_epochs"]):
    B.epsilon = config["epsilon_start"] if i == 0 else config["epsilon"]

    state, _ = train(
        state,
        model,
        opt,
        objective,
        gen_train,
        fix_noise=fix_noise,
    )

    # The epoch is done. Now evaluate.
    state, val = eval(state, model, objective, gen_cv())

    # Save current model.
    
    torch.save(
        {
            "weights": model.state_dict(),
            "objective": val,
            "epoch": i + 1,
        },
        latest_model_path,
    )

    # Check if the model is the new best. If so, save it.
    if val > best_eval_lik:
        print("New best model!")
        best_eval_lik = val
        torch.save(
            {
                "weights": model.state_dict(),
                "objective": val,
                "epoch": i + 1,
            },
            best_model_path,
        )

    # Visualise a few predictions by the model.
    for j in range(2):
        visualise(
            model,
            gen_cv(),
            path=f"{train_plot_dir}/train-epoch-{i + 1:03d}-{j + 1}.pdf",
            config=config,
        )
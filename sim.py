# %%
from functools import partial
import warnings
from tqdm import tqdm

import neuralprocesses.torch as nps
import numpy as np
import torch
import lab as B
import wandb

from utils import (
    ensure_exists,
    load_weights,
    save_model,
    get_exp_dir,
    get_paths,
    get_exp_dir_sim,
)
from train import train, setup, evaluate
from models.convgnp import construct_convgnp
from plot import visualise_1d
from dataclasses import asdict

from runspec import spec

# %%
exp_dir = get_exp_dir_sim(spec)
train_plot_dir, best_model_path, latest_model_path, sim_model_dir = get_paths(exp_dir)

ensure_exists(train_plot_dir)
ensure_exists(sim_model_dir)
print("Working dir: ", exp_dir)

device = spec.device
B.set_global_device(device)


if spec.out.wandb:
    wandb.init(
        project="thesis",
        config=asdict(spec),
        name=f"sim {spec.data.lengthscale}",
    )

gen_train, gen_cv, gens_eval = setup(spec.data, spec.device)


model = construct_convgnp(
    points_per_unit=spec.data.ppu,
    dim_x=spec.data.dim_x,
    dim_yc=(1,) * spec.data.dim_y,
    dim_yt=spec.data.dim_y,
    likelihood="het",
    conv_arch=spec.model.arch,
    unet_channels=spec.model.unet.channels,
    unet_strides=spec.model.unet.strides,
    conv_channels=spec.model.conv.channels,
    conv_layers=spec.model.num_layers,
    conv_receptive_field=spec.model.conv.receptive_field,
    margin=spec.model.margin,
    encoder_scales=spec.model.encoder_scales,
    encoder_scales_learnable=spec.model.encoder_scales_learnable,
    transform=spec.model.transform,
    affine=spec.model.affine,
    freeze_affine=spec.model.freeze_affine,
    residual=spec.model.residual,
    kernel_size=spec.model.kernel_size,
    unet_resize_convs=spec.model.unet.resize_convs,
    unet_resize_conv_interp_method=spec.model.unet.resize_conv_interp_method,
)

print(model)


objective = partial(
    nps.loglik,
    num_samples=spec.data.num_samples,
    normalise=spec.model.normalise_obj,
)
# %%
opt = torch.optim.Adam(model.parameters(), spec.opt.lr)

state = B.create_random_state(torch.float32, seed=0)

best_eval_lik = -np.infty

pbar = tqdm(range(spec.opt.num_epochs))

for i in pbar:
    state, train_lik, true_train_lik = train(
        state,
        model,
        opt,
        objective,
        gen_train,
    )

    # The epoch is done. Now evaluate.
    state, val_lik, true_val_lik = evaluate(state, model, objective, gen_cv())

    measures = {
        "train_lik": train_lik,
        "val_lik": val_lik,
    }

    pbar.set_postfix(measures)
    if spec.out.wandb:
        measures["true_train_lik"] = true_train_lik
        measures["true_val_lik"] = true_val_lik
        wandb.log(measures)

    # Save current model.
    save_model(model, true_val_lik, i + 1, latest_model_path)

    # Check if the model is the new best. If so, save it.
    if true_val_lik > best_eval_lik:
        best_eval_lik = true_val_lik
        save_model(model, true_val_lik, i + 1, best_model_path)

    # Visualise a few predictions by the model.
    gcv = gen_cv()
    for j in range(2):
        visualise_1d(
            spec,
            model,
            gcv,
            path=f"{train_plot_dir}/train-epoch-{i + 1:03d}-{j + 1}.pdf",
        )

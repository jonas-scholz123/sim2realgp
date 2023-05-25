# %%
from utils import get_exp_dir_sim2real, get_paths, load_weights
from train import evaluate, setup
import torch
import numpy as np
from functools import partial
from tqdm import tqdm

import neuralprocesses.torch as nps

from config import sim2real_spec as spec, config
from models.convgnp import construct_convgnp
from finetuners.tuner_types import TunerType

import lab as B

model = construct_convgnp(
    points_per_unit=spec.real.ppu,
    dim_x=spec.real.dim_x,
    dim_yc=(1,) * spec.real.dim_y,
    dim_yt=spec.real.dim_y,
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

objective = partial(
    nps.loglik,
    num_samples=spec.real.num_samples,
    normalise=spec.model.normalise_obj,
)

l = config["lengthscales_real"][0]
num_tasks = config["real_nums_tasks_train"][0]
tuner = config["tuners"][0]

ls = [0.05, 0.1, 0.2]
nums_tasks = [2**4, 2**8, 2**10]
spec.real.num_tasks_val = 2**6
tuners = [TunerType.film, TunerType.freeze, TunerType.naive]

best_tuners = np.zeros((len(ls), len(nums_tasks)))
best_liks = np.zeros((len(ls), len(nums_tasks)))
for i, l in enumerate(tqdm(ls)):
    for j, num_tasks in enumerate(tqdm(nums_tasks)):
        best_lik = -float("inf")
        best_tuner = None

        for tuner in tuners:
            state = B.create_random_state(torch.float32, seed=0)
            spec.real.lengthscale = l
            spec.real.num_tasks_train = num_tasks
            spec.tuner = tuner
            _, _, gen = setup(spec.real, spec.device)

            sim_exp_dir, tuned_exp_dir = get_exp_dir_sim2real(spec)
            train_plot_dir, best_model_path, _, model_dir = get_paths(tuned_exp_dir)
            model, _ = load_weights(model, best_model_path)
            model = model.to(spec.device)
            state, val_lik, true_val_lik = evaluate(state, model, objective, gen())

            if val_lik > best_lik:
                best_lik = val_lik
                best_tuner = tuner

        best_tuners[i, j] = best_tuner.value
        best_liks[i, j] = best_lik

# %%
best_liks

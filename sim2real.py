# %%
import torch
import neuralprocesses.torch as nps
from dataclasses import asdict
import lab as B
import wandb
from tqdm import tqdm
from config import config
from plot import visualise, visualise_1d
from train import train, evaluate, setup
from functools import partial
from utils import (
    save_model,
    load_weights,
    ensure_exists,
    get_exp_dir,
    get_paths,
    get_exp_dir_sim2real,
)
from models.convgnp import construct_convgnp
from finetuners.get_tuner import get_tuner

from runspec import Sim2RealSpec
from config import sim2real_spec as spec
from dataclasses import replace


def sim2real(spec: Sim2RealSpec):
    device = spec.device
    B.set_global_device(device)
    # simulator only.
    sim_exp_dir = get_exp_dir(config)

    # simulator pretrained + finetuned on real data of given lengthscale.
    # tuned_exp_dir = get_exp_dir(config, spec.real.lengthscale, num_tasks, tuner_type)
    sim_exp_dir, tuned_exp_dir = get_exp_dir_sim2real(spec)
    print(f"Tuned: {tuned_exp_dir}")

    train_plot_dir, best_model_path, latest_model_path, model_dir = get_paths(
        tuned_exp_dir
    )

    ensure_exists(model_dir)
    ensure_exists(train_plot_dir)

    gen_train, gen_cv, gens_eval = setup(spec.real, spec.device)

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

    model = model.to(spec.device)
    # %%

    best_pretrained_path = get_paths(sim_exp_dir)[1]
    print(f"Loading best model from {best_pretrained_path}")
    model = load_weights(model, best_pretrained_path)

    batches = list(gen_train.epoch())
    batches = batches * (spec.opt.epoch_size // (len(batches) * spec.opt.batch_size))
    print(f"Using {len(batches)} batches.")

    tuner = get_tuner(tuner_type)(model, objective, torch.optim.Adam, spec.opt.lr)
    state = B.create_random_state(torch.float32, seed=0)

    if spec.real.inf_tasks:
        num_tasks = "inf"

    if spec.out.wandb:
        spec_dir = asdict(spec)
        run = wandb.init(
            project="thesis",
            config=spec_dir,
            name=f"tune {spec.sim.lengthscale} -> {spec.real.lengthscale}, {num_tasks} tasks",
            reinit=True,
        )

    print(f"Tuning using {tuner}")

    if spec.out.wandb:
        state, val_lik, true_val_lik = evaluate(state, tuner.model, objective, gen_cv())
        state, train_lik, true_train_lik = evaluate(
            state, tuner.model, objective, gen_train
        )

        measures = {
            "train_lik": train_lik,
            "val_lik": val_lik,
            "true_train_lik": true_train_lik,
            "true_val_lik": true_val_lik,
        }
        run.log(measures)

    best_eval_lik = -float("inf")

    pbar = tqdm(range(spec.opt.num_epochs))
    for i in pbar:
        train_lik = 0
        true_train_lik = 0
        for batch in batches:
            state, batch_lik, batch_true_lik = tuner.train_on_batch(batch, state)
            train_lik += batch_lik / len(batches)
            true_train_lik += batch_true_lik / len(batches)

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
            run.log(measures)

        # Save current model.
        save_model(model, val_lik, i + 1, latest_model_path)

        # Check if the model is the new best. If so, save it.
        if val_lik > best_eval_lik:
            best_eval_lik = val_lik
            save_model(model, val_lik, i + 1, best_model_path)

        if spec.out.visualise:
            # Visualise a few predictions by the model.
            for j in range(2):
                visualise_1d(
                    spec.out,
                    spec.real,
                    model,
                    gen_cv(),
                    path=f"{train_plot_dir}/train-epoch-{i + 1:03d}-{j + 1}.pdf",
                )

        if spec.real.inf_tasks:
            # Get a fresh batch of tasks.
            batches = list(gen_train.epoch())


if __name__ == "__main__":
    lengthscales = config["lengthscales_real"]
    nums_tasks = config["real_nums_tasks_train"]
    tuner_types = config["tuners"]
    for lengthscale in lengthscales:
        for num_tasks in nums_tasks:
            for tuner_type in tuner_types:
                spec.real.lengthscale = lengthscale
                spec.real.num_tasks_train = num_tasks
                spec.tuner = tuner_type
                sim2real(spec)

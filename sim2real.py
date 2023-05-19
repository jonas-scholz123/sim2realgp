# %%
import torch
import neuralprocesses.torch as nps
import lab as B
import wandb
from tqdm import tqdm
from config import config
from plot import visualise
from train import train, evaluate, setup
from functools import partial
from utils import save_model, load_weights, ensure_exists, get_exp_dir, get_paths
from models.convgnp import construct_convgnp
from finetuners.get_tuner import get_tuner


def sim2real(tuner_type, real_lengthscale, num_tasks):
    device = config["device"]
    B.set_global_device(device)
    # simulator only.
    sim_exp_dir = get_exp_dir(config)

    # simulator pretrained + finetuned on real data of given lengthscale.
    tuned_exp_dir = get_exp_dir(config, real_lengthscale, num_tasks, tuner_type)
    print(f"Tuned: {tuned_exp_dir}")

    train_plot_dir, best_model_path, latest_model_path, model_dir = get_paths(
        tuned_exp_dir
    )

    ensure_exists(model_dir)
    ensure_exists(train_plot_dir)

    gen_train, gen_cv, gens_eval = setup(
        config,
        num_tasks_train=num_tasks,
        num_tasks_val=config["real_num_tasks_val"],
        lengthscale=real_lengthscale,
    )

    model = construct_convgnp(
        points_per_unit=config["points_per_unit"],
        dim_x=config["dim_x"],
        dim_yc=(1,) * config["dim_y"],
        dim_yt=config["dim_y"],
        likelihood="het",
        conv_arch=config["arch"],
        unet_channels=config["unet_channels"],
        unet_strides=config["unet_strides"],
        conv_channels=config["conv_channels"],
        conv_layers=config["num_layers"],
        conv_receptive_field=config["conv_receptive_field"],
        margin=config["margin"],
        encoder_scales=config["encoder_scales"],
        encoder_scales_learnable=config["encoder_scales_learnable"],
        transform=config["transform"],
        affine=config["affine"],
        freeze_affine=config["freeze_affine"],
        residual=config["residual"],
        kernel_size=config["kernel_size"],
        unet_resize_convs=config["unet_resize_convs"],
        unet_resize_conv_interp_method=config["unet_resize_conv_interp_method"],
    )

    objective = partial(
        nps.loglik,
        num_samples=config["num_samples"],
        normalise=config["normalise_obj"],
    )

    model = model.to(config["device"])
    # %%

    best_pretrained_path = get_paths(sim_exp_dir)[1]
    print(f"Loading best model from {best_pretrained_path}")
    model = load_weights(model, best_pretrained_path)

    batches = list(gen_train.epoch())
    batches = batches * (config["epoch_size"] // (len(batches) * config["batch_size"]))
    print(f"Using {len(batches)} batches.")

    tuner = get_tuner(tuner_type)(
        model, objective, torch.optim.Adam, config["tune_rate"]
    )
    state = B.create_random_state(torch.float32, seed=0)

    sim_l = config["lengthscale_sim"]

    if config["real_inf_tasks"]:
        num_tasks = "inf"

    if config["wandb"]:
        run = wandb.init(
            project="thesis",
            config={
                "stage": "tuning",
                "arch": config["arch"],
                "sim_lengthscale": config["lengthscale_sim"],
                "real_lengthscale": real_lengthscale,
                "real_num_tasks": num_tasks,
                "tuner": tuner.name(),
            },
            name=f"tune {sim_l} -> {real_lengthscale}, {num_tasks} tasks",
            reinit=True,
        )

    print(f"Tuning using {tuner}")

    if config["wandb"]:
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

    pbar = tqdm(range(config["num_epochs"]))
    for i in pbar:
        B.epsilon = config["epsilon_start"] if i == 0 else config["epsilon"]

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
        if config["wandb"]:
            measures["true_train_lik"] = true_train_lik
            measures["true_val_lik"] = true_val_lik
            run.log(measures)

        # Save current model.
        save_model(model, val_lik, i + 1, latest_model_path)

        # Check if the model is the new best. If so, save it.
        if val_lik > best_eval_lik:
            best_eval_lik = val_lik
            save_model(model, val_lik, i + 1, best_model_path)

        if config["visualise"]:
            # Visualise a few predictions by the model.
            for j in range(2):
                visualise(
                    model,
                    gen_cv(),
                    path=f"{train_plot_dir}/train-epoch-{i + 1:03d}-{j + 1}.pdf",
                    config=config,
                )

        if config["real_inf_tasks"]:
            # Get a fresh batch of tasks.
            batches = list(gen_train.epoch())


if __name__ == "__main__":
    lengthscales = config["lengthscales_real"]
    nums_tasks = config["real_nums_tasks_train"]
    tuner_types = config["tuners"]
    for lengthscale in lengthscales:
        for num_tasks in nums_tasks:
            for tuner_type in tuner_types:
                sim2real(tuner_type, lengthscale, num_tasks)

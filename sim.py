#%%
from functools import partial
import warnings
from tqdm import tqdm

import neuralprocesses.torch as nps
import numpy as np
import torch
import lab as B

from utils import ensure_exists, load_weights, save_model, get_exp_dir, get_paths
from train import train, setup, evaluate
from models.convgnp import construct_convgnp
from plot import visualise
from config import config
#%%

exp_dir = get_exp_dir(config)
train_plot_dir, best_model_path, latest_model_path, sim_model_dir = get_paths(exp_dir)

ensure_exists(train_plot_dir)
ensure_exists(sim_model_dir)
print("Working dir: ", exp_dir)

device = config["device"]
B.set_global_device(device)

gen_train, gen_cv, gens_eval = setup(
    config,
    num_tasks_train=config["sim_num_tasks_train"],
    num_tasks_val=config["sim_num_tasks_val"],
    lengthscale=config["lengthscale_sim"]
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

for i in tqdm(range(config["num_epochs"])):
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
    state, val = evaluate(state, model, objective, gen_cv())

    # Save current model.
    save_model(model, val, i + 1, latest_model_path)
    
    # Check if the model is the new best. If so, save it.
    if val > best_eval_lik:
        print("New best model!")
        best_eval_lik = val
        save_model(model, val, i + 1, best_model_path)

    # Visualise a few predictions by the model.
    for j in range(2):
        visualise(
            model,
            gen_cv(),
            path=f"{train_plot_dir}/train-epoch-{i + 1:03d}-{j + 1}.pdf",
            config=config,
        )
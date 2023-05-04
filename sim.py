#%%
from functools import partial
import warnings

import neuralprocesses.torch as nps
import numpy as np
import torch
import lab as B
from matrix.util import ToDenseWarning

from utils import ensure_exists
from train import train, setup, evaluate
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

device = config["device"]
B.set_global_device(device)

gen_train, gen_cv, gens_eval = setup(
    config,
    num_tasks_train=config["sim_num_tasks_train"],
    num_tasks_val=config["sim_num_tasks_val"],
    lengthscale=config["lengthscale_sim"]
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
    state, val = evaluate(state, model, objective, gen_cv())

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
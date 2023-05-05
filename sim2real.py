#%%
import torch
import neuralprocesses.torch as nps
import lab as B
import wandb
from tqdm import tqdm

from config import config
from plot import visualise
from train import train, evaluate, setup
from functools import partial
from finetuners.naive_tuner import NaiveTuner
from utils import save_model, load_weights, ensure_exists, get_exp_dir, get_paths

# True lengthscale is 0.25.
# TODO: move to config.
lengthscales = [0.1, 0.2]
lengthscale = 0.1

# simulator only.
sim_exp_dir = get_exp_dir(config)

# simulator pretrained + finetuned on real data of given lengthscale.
tuned_exp_dir = get_exp_dir(config, lengthscale, config["real_num_tasks_train"])
print(f"Tuned: {tuned_exp_dir}")

train_plot_dir, best_model_path, latest_model_path, model_dir = get_paths(tuned_exp_dir)

ensure_exists(model_dir)
ensure_exists(train_plot_dir)

gen_train, gen_cv, gens_eval = setup(
    config,
    num_tasks_train=config["real_num_tasks_train"],
    num_tasks_val=config["real_num_tasks_val"],
    lengthscale=lengthscale
)

model = nps.construct_convgnp(
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

model = model.to(config["device"])
model.decoder
#%%

best_pretrained_path = get_paths(sim_exp_dir)[1]
print(f"Loading best model from {best_pretrained_path}")
model = load_weights(model, best_pretrained_path)

# Don't want to generate new tasks, make one epoch and reuse.
batches = list(gen_train.epoch())
#TODO: different LR for tuning
tuner = NaiveTuner(model, objective, torch.optim.Adam, config["rate"])
state = B.create_random_state(torch.float32, seed=0)

n_tasks = config["real_num_tasks_train"]
sim_l = config["lengthscale_sim"]

wandb.init(
    project="thesis",
    config={
        "stage": "tuning",
        "sim_lengthscale": config["lengthscale_sim"],
        "real_lengthscale": lengthscale,
        "real_num_tasks": n_tasks,
        "tuner": tuner.name(),
    },
    name=f"tune {sim_l} -> {lengthscale}, {n_tasks} tasks"
)

print(f"Tuning using {tuner}")

state, val = evaluate(state, tuner.model, objective, gen_cv())

best_eval_lik = -float("inf")

for i in range(config["num_epochs"]):
    B.epsilon = config["epsilon_start"] if i == 0 else config["epsilon"]

    print("epoch: ", i)
    train_lik = 0
    for batch in tqdm(batches):
        state, batch_lik = tuner.train_on_batch(batch, state)
        train_lik -= batch_lik / len(batches)

    # The epoch is done. Now evaluate.
    state, val_lik = evaluate(state, model, objective, gen_cv())
    wandb.log({"train_lik": train_lik, "val_likelihood": val_lik})

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
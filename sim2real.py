import torch
import neuralprocesses.torch as nps
import lab as B
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
lengthscale = 0.2

# simulator only.
sim_exp_dir = get_exp_dir(config)

# simulator pretrained + finetuned on real data of given lengthscale.
tuned_exp_dir = get_exp_dir(config, lengthscale)

train_plot_dir, best_model_path, latest_model_path, model_dir = get_paths(tuned_exp_dir)

ensure_exists(model_dir)
ensure_exists(train_plot_dir)

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

model = model.to(config["device"])

print("Loading best model")
best_pretrained_path = get_paths(sim_exp_dir)[1]
model = load_weights(model, best_pretrained_path)

# Don't want to generate new tasks, make one epoch and reuse.
batches = list(gen_train.epoch())
#TODO: different LR for tuning
tuner = NaiveTuner(model, objective, torch.optim.Adam, config["rate"])
state = B.create_random_state(torch.float32, seed=0)

print(f"Tuning using {tuner}")

state, val = evaluate(state, tuner.model, objective, gen_cv())

best_eval_lik = -float("inf")

for i in range(config["num_epochs"]):
    B.epsilon = config["epsilon_start"] if i == 0 else config["epsilon"]

    print("epoch: ", i)
    for batch in tqdm(batches):
        state = tuner.train_on_batch(batch, state)

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
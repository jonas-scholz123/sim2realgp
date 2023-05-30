# %%
from typing import List

from attr import dataclass
from utils import get_exp_dir_sim, get_exp_dir_sim2real, get_paths, load_weights
from train import evaluate, setup
import torch
import numpy as np
import matplotlib.patches as mpatches
from functools import partial
from itertools import product
from dataclasses import replace, asdict
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

import neuralprocesses.torch as nps

from config import sim2real_spec as spec, config, sim_spec
from models.convgnp import construct_convgnp
from finetuners.tuner_types import TunerType
from runspec import Sim2RealSpec, SimRunSpec
import matplotlib.pyplot as plt
import seaborn as sns

import lab as B

l = config["lengthscales_real"][0]
num_tasks = config["real_nums_tasks_train"][0]
tuner = config["tuners"][0]

ls = [0.05, 0.1, 0.2]
nums_tasks = [2**4, 2**6, 2**8, 2**10]
spec.real.num_tasks_val = 2**6
tuners = [TunerType.film, TunerType.naive]
seeds = range(10, 20)


def modify_s2rspec(spec, l, num_tasks, tuner, seed) -> Sim2RealSpec:
    spec = deepcopy(spec)
    spec.real.lengthscale = l
    spec.real.num_tasks_train = num_tasks
    spec.tuner = tuner
    spec.real.train_seed = seed
    return spec


def gen_s2rspecs(spec, ls, nums_tasks, tuners, seeds) -> List[Sim2RealSpec]:
    specs = []
    for l, num_tasks, tuner, seed in product(ls, nums_tasks, tuners, seeds):
        specs.append(modify_s2rspec(spec, l, num_tasks, tuner, seed))
    return specs


def gen_simspecs(spec: SimRunSpec, ls) -> List[SimRunSpec]:
    specs = []
    for l in ls:
        s = deepcopy(spec)
        s.data.lengthscale = l
        specs.append(s)
    return specs


# %%

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
            model, val_lik = load_weights(model, best_model_path)
            model = model.to(spec.device)
            # state, _, true_val_lik = evaluate(state, model, objective, gen())

            if val_lik > best_lik:
                best_lik = val_lik
                best_tuner = tuner

        best_tuners[i, j] = best_tuner.value
        best_liks[i, j] = best_lik

# %%

specs = gen_s2rspecs(spec, ls, nums_tasks, tuners, seeds)

records = []
for spec in specs:
    try:
        sim_exp_dir, tuned_exp_dir = get_exp_dir_sim2real(spec)
        _, best_model_path, _, _ = get_paths(tuned_exp_dir)
        _, cv_lik = load_weights(model, best_model_path, lik_only=True)

        record = {
            "lengthscale": spec.real.lengthscale,
            "num_tasks": spec.real.num_tasks_train,
            "tuner": spec.tuner,
            "seed": spec.real.train_seed,
            "cv_lik": cv_lik,
        }
        records.append(record)
    except FileNotFoundError:
        continue

df = pd.DataFrame(records)

diffs = np.zeros((len(ls), len(nums_tasks)))
for i, l in enumerate(tqdm(ls)):
    for j, num_tasks in enumerate(tqdm(nums_tasks)):
        nl_df = df[(df["num_tasks"] == num_tasks) & (df["lengthscale"] == l)]
        film = nl_df[nl_df["tuner"] == TunerType.film]["cv_lik"].mean()
        naive = nl_df[nl_df["tuner"] == TunerType.naive]["cv_lik"].mean()
        diffs[i, j] = film - naive

sns.heatmap(
    diffs,
    xticklabels=nums_tasks,
    yticklabels=ls,
    cmap="RdBu",
    vmin=-0.04,
    vmax=0.04,
    annot=True,
)

# %%
# df[(df["lengthscale"] == 0.1) & (df["num_tasks"] == 256)]
# df["tuner"] = df["tuner"].astype("string")
# df.groupby(["lengthscale", "num_tasks", "tuner"]).mean()

df = df[(df["num_tasks"] == 16) & (df["lengthscale"] == 0.05)]
df
# %%

a = df[df["tuner"] == TunerType.film]["cv_lik"].reset_index(drop=True)
b = df[df["tuner"] == TunerType.naive]["cv_lik"].reset_index(drop=True)
delta = b - a
# df[df["tuner"] == TunerType.film]["cv_lik"].mean()

# %%
delta.std()
# %%
df["tuner"] = df["tuner"].astype("string")
df.groupby(["lengthscale", "num_tasks", "tuner"]).mean()
# %%

records = []

for l in [0.1, 0.2]:
    s = replace(sim_spec)
    s.data.lengthscale = l
    exp_dir = get_exp_dir_sim(s)
    _, best_model_path, _, _ = get_paths(exp_dir)
    _, val_lik = load_weights(model, best_model_path, lik_only=True)

    records.append(
        {
            "lengthscale": l,
            "val_lik": val_lik,
        }
    )

baselines = pd.DataFrame(records)
baselines


# %%
@dataclass
class Record:
    lengthscale: float
    num_tasks: int
    tuner: TunerType
    seed: int
    val_lik: float
    true_val_lik: float


class Evaluator:
    def __init__(self, spec: Sim2RealSpec, val_tasks=2**6) -> None:
        self.model = construct_convgnp(
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

        self.objective = partial(
            nps.loglik,
            num_samples=spec.real.num_samples,
            normalise=spec.model.normalise_obj,
        )

        self.df = pd.DataFrame(
            columns=[
                "lengthscale",
                "num_tasks",
                "tuner",
                "seed",
                "val_lik",
                "true_val_lik",
            ]
        )

        self.records = []
        self.device = spec.device
        self.val_tasks = val_tasks
        self.path = "./outputs/results.csv"

    def _eval(self, path: str, gen):
        try:
            state = B.create_random_state(torch.float32, seed=0)
            model, _ = load_weights(self.model, path)
            model = model.to(self.device)
            state, val_lik, true_val_lik = evaluate(state, model, self.objective, gen())
            return val_lik, true_val_lik
        except FileNotFoundError as e:
            return -1, -1

    def eval_s2r(self, spec: Sim2RealSpec):
        spec.real.num_tasks_val = self.val_tasks
        _, _, gen = setup(spec.real, spec.device)
        _, tuned_exp_dir = get_exp_dir_sim2real(spec)
        _, best_model_path, _, _ = get_paths(tuned_exp_dir)

        lik, true_lik = self._eval(best_model_path, gen)

        if lik != -1:
            self._add_record(
                Record(
                    spec.real.lengthscale,
                    spec.real.num_tasks_train,
                    spec.tuner,
                    spec.real.train_seed,
                    lik,
                    true_lik,
                )
            )
        return lik, true_lik

    def eval_baseline(self, spec: Sim2RealSpec):
        # Generator produces "real" data.
        spec.real.num_tasks_val = self.val_tasks
        _, _, gen = setup(spec.real, spec.device)

        sim_spec = SimRunSpec(
            spec.device,
            spec.out,
            spec.sim,
            spec.model,
            spec.opt,
            spec.dirs,
        )

        # We load untuned model.
        exp_dir = get_exp_dir_sim(sim_spec)
        _, best_model_path, _, _ = get_paths(exp_dir)

        lik, true_lik = self._eval(best_model_path, gen)

        if lik != -1:
            self._add_record(
                Record(
                    spec.real.lengthscale,
                    0,  # No data, i.e. 0 shot.
                    None,  # No tuner
                    spec.real.train_seed,
                    lik,
                    true_lik,
                )
            )
        return lik, true_lik

    def eval_sim(self, spec: SimRunSpec):
        spec.data.num_tasks_val = self.val_tasks
        _, _, gen = setup(spec.data, spec.device)
        exp_dir = get_exp_dir_sim(spec)
        _, best_model_path, _, _ = get_paths(exp_dir)

        lik, true_lik = self._eval(best_model_path, gen)

        if lik != -1:
            self._add_record(
                Record(spec.data.lengthscale, float("inf"), None, 10, lik, true_lik)
            )
        return lik, true_lik

    def _add_record(self, record: Record):
        idx = 0 if self.df.empty else self.df.index.max() + 1
        self.df.loc[idx] = record.__dict__

    def save(self):
        self.df.to_csv(self.path, index=False)

    def load(self):
        self.df = pd.read_csv(self.path)

    def at(self, l=None, num=None, tuner=None):
        df = self.df

        if l is not None:
            df = df[df["lengthscale"] == l]
        if num is not None:
            df = df[df["num_tasks"] == num]
            if num == float("inf"):
                return df

        if tuner is not None:
            df = df[df["tuner"] == str(tuner)]

        return df


e = Evaluator(spec, 2**8)
# e.load()
specs = gen_s2rspecs(spec, ls, nums_tasks, tuners, seeds)
simspecs = gen_simspecs(sim_spec, ls)

for s in tqdm(specs):
    e.eval_s2r(s)

    if (
        s.tuner == TunerType.naive
        and s.real.num_tasks_train == nums_tasks[0]
        and s.real.train_seed == seeds[0]
    ):
        e.eval_baseline(s)

for ss in tqdm(simspecs):
    e.eval_sim(ss)


# %%
e.save()
e.load()
# %%
ls = [0.05, 0.1, 0.2]
fig, axs = plt.subplots(1, len(ls), figsize=(12, 4))

c1 = "C0"
c2 = "C1"

for l, ax in zip(ls, axs):
    means, stds, labels, colors = [], [], [], []
    ax.set_title(f"0.25 $\\rightarrow$ {l}")

    if l != 0.05:
        # Add 0-shot baseline:
        means.append(float(e.at(l=l, num=0)["val_lik"]))
        stds.append(0)
        labels.append("0 Shot")
        colors.append("grey")

    for num in [16, 64, 256]:
        for tuner in [TunerType.naive, TunerType.film]:
            liks = e.at(l, num, tuner)["val_lik"]
            means.append(liks.mean())
            stds.append(1.96 * liks.std())
            labels.append(f"{tuner.name}_{num}")
            colors.append(c1 if tuner == TunerType.naive else c2)

    inf_df = e.at(l=l, num=float("inf"))

    # Add infinite data baseline:
    inf_lik = float(inf_df["val_lik"])
    means.append(inf_lik)
    stds.append(0)
    labels.append("naive_inf")
    colors.append("grey")

    # Add true likelihood baseline:
    true_lik = float(inf_df["true_val_lik"])
    ymin, ymax = min(means) * 1.05, true_lik * 0.95

    xmin, xmax = -1, len(labels)
    bars = ax.bar(labels, means, yerr=stds)

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    truth = ax.hlines(true_lik, xmin, xmax, colors="black", linestyles="--", label="GP")

    naive_patch = mpatches.Patch(color=c1, label="Naive")
    film_patch = mpatches.Patch(color=c2, label="FiLM")

    ax.legend(handles=[truth, naive_patch, film_patch])

    # %%

    e.df[e.df["num_tasks"] == 0]
    e.at(num=0)

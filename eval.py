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
import matplotlib.pyplot as plt
import seaborn as sns

import neuralprocesses.torch as nps

from config import sim2real_spec as spec, config, sim_spec
from models.convgnp import construct_convgnp
from finetuners.tuner_types import TunerType
from runspec import Sim2RealSpec, SimRunSpec
from plot import save

import lab as B

l = config["lengthscales_real"][0]
num_tasks = config["real_nums_tasks_train"][0]
tuner = config["tuners"][0]

spec.real.num_tasks_val = 2**10

experiment = "multiscale"

if experiment == "multiscale":
    ls = [0.05, 0.1, 0.2]
    noises = [0.05]
    nums_tasks = [2**4, 2**6, 2**8, "inf"]
    spec.sim.lengthscale = (0.25, 0.5)
    path = "./outputs/results_multiscale.csv"
elif experiment == "lengthscale":
    spec.sim.lengthscale = 0.25
    ls = [0.05, 0.1, 0.2]
    noises = [0.05]
    nums_tasks = [2**4, 2**6, 2**8, "inf"]
    path = "./outputs/results.csv"
elif experiment == "noise":
    spec.sim.lengthscale = 0.25
    ls = [0.25]
    noises = [0.0125, 0.025, 0.1, 0.2]
    nums_tasks = [2**4, 2**6, 2**8, 2**10, "inf"]
    path = "./outputs/results_noise.csv"
tuners = [TunerType.film, TunerType.naive]
seeds = range(10, 20)


def modify_s2rspec(spec, l, noise, num_tasks, tuner, seed) -> Sim2RealSpec:
    spec = deepcopy(spec)
    spec.real.lengthscale = l
    spec.real.noise = noise
    if num_tasks != "inf":
        spec.real.num_tasks_train = num_tasks
    else:
        spec.real.num_tasks_train = 2**10
        spec.real.inf_tasks = True
    spec.tuner = tuner
    spec.real.train_seed = seed
    return spec


def gen_s2rspecs(spec, ls, noises, nums_tasks, tuners, seeds) -> List[Sim2RealSpec]:
    specs = []
    for l, noise, num_tasks, tuner, seed in product(
        ls, noises, nums_tasks, tuners, seeds
    ):
        specs.append(modify_s2rspec(spec, l, noise, num_tasks, tuner, seed))
    return specs


def gen_simspecs(spec: SimRunSpec, ls, noises) -> List[SimRunSpec]:
    specs = []
    for l in ls:
        s = deepcopy(spec)
        s.data.lengthscale = l
        specs.append(s)
    return specs


# %%
@dataclass
class Record:
    lengthscale: float
    noise: float
    num_tasks: int
    tuner: TunerType
    seed: int
    val_lik: float
    true_val_lik: float


class Evaluator:
    def __init__(
        self, spec: Sim2RealSpec, val_tasks=2**6, path="./outputs/results.csv"
    ) -> None:
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
                "noise",
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
        self.path = path

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

        num_tasks = spec.real.num_tasks_train if not spec.real.inf_tasks else np.inf

        if lik != -1:
            self._add_record(
                Record(
                    spec.real.lengthscale,
                    spec.real.noise,
                    num_tasks,
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
                    spec.real.noise,
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
                Record(
                    spec.data.lengthscale,
                    spec.data.noise,
                    float("inf"),
                    None,
                    10,
                    lik,
                    true_lik,
                )
            )
        return lik, true_lik

    def _add_record(self, record: Record):
        idx = 0 if self.df.empty else self.df.index.max() + 1
        self.df.loc[idx] = record.__dict__

    def save(self):
        self.df.to_csv(self.path, index=False)

    def load(self):
        self.df = pd.read_csv(self.path)

    def at(self, l=None, noise=None, num=None, tuner=None):
        df = self.df

        if l is not None:
            df = df[df["lengthscale"] == l]
        if noise is not None:
            df = df[df["noise"] == noise]
        if num is not None:
            df = df[df["num_tasks"] == num]
        if tuner is not None:
            # saving/loading turns obj into string. Catch both cases.
            df = df[(df["tuner"] == str(tuner)) | (df["tuner"] == tuner)]

        return df


# %%
e = Evaluator(spec, 2**10, path)
# e.load()
# %%
specs = gen_s2rspecs(spec, ls, noises, nums_tasks, tuners, seeds)

for s in tqdm(specs):
    e.eval_s2r(s)

    if (
        s.tuner == TunerType.naive
        and s.real.num_tasks_train == nums_tasks[0]
        and s.real.train_seed == seeds[0]
    ):
        e.eval_baseline(s)

simspecs = gen_simspecs(sim_spec, ls, noises)
for ss in tqdm(simspecs):
    e.eval_sim(ss)


# %%
e.save()
e.load()
e.df


# %%
def gap_plots(e, name=None):
    fig, axs = plt.subplots(1, len(ls), figsize=(12, 4))

    c1 = "C0"
    c2 = "C1"

    # ignore inf entry
    nums = nums_tasks[:-1]

    for l, ax in zip(ls, axs):
        means, stds, labels, colors = [], [], [], []
        ax.set_title(f"{spec.sim.lengthscale} (sim) $\\rightarrow$ {l} (real)")

        # if l != 0.05:
        #    # Add 0-shot baseline:
        #    means.append(float(e.at(l=l, num=0)["val_lik"]))
        #    stds.append(0)
        #    labels.append("0 Shot")
        #    colors.append("grey")

        x1 = np.linspace(0, 4, len(nums))
        x2 = x1 + 0.2
        for x, color, tuner in zip(
            [x1, x2], [c1, c2], [TunerType.naive, TunerType.film]
        ):
            means, stds, labels, colors = [], [], [], []
            for num in nums:
                liks = e.at(l, 0.05, num, tuner)["val_lik"]
                means.append(liks.mean())
                stds.append(1.96 * liks.std() / np.sqrt(len(liks)))
                labels.append(f"{tuner.name}_{num}")
                colors.append(c1 if tuner == TunerType.naive else c2)
            ax.errorbar(x, means, fmt=".", yerr=stds, ecolor=color, color=color)

        inf_df = e.at(l=l, num=float("inf"))

        # Add true likelihood baseline:
        true_lik = float(inf_df["true_val_lik"])
        ymin, ymax = min(means), true_lik * 0.95
        if l != 0.05:
            zero_shot = float(e.at(l=l, num=0)["val_lik"])
            ymin = 1.05 * min(ymin, zero_shot)
            # ymin *= 1.05
        else:
            ymin *= 1.05

        xmin, xmax = x1.min() - 1, x2.max() + 1
        # bars = ax.bar(labels, means, yerr=stds)
        # bp = ax.boxplot(all_liks, patch_artist=True, whis=10000)

        # for patch, color in zip(bp["boxes"], colors):
        #    patch.set_facecolor(color)

        # for marker, color in zip(markers, colors):
        #    print(marker)
        #    marker.set_color(color)

        # for bar, color in zip(bars, colors):
        #    bar.set_color(color)

        # ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        truth = ax.hlines(
            true_lik, xmin, xmax, colors="black", linestyles="--", label="GP"
        )

        # Add infinite data baseline:
        inf_lik = float(inf_df["val_lik"])
        inf = ax.hlines(
            inf_lik, xmin, xmax, colors="green", linestyles="--", label="$\\infty$ data"
        )

        naive_patch = mpatches.Patch(color=c1, label="Naive")
        film_patch = mpatches.Patch(color=c2, label="FiLM")

        if l != 0.05:
            # Add 0-shot baseline:
            zero = ax.hlines(
                zero_shot, xmin, xmax, colors="red", linestyles="--", label="0-shot"
            )
            ax.legend(handles=[truth, inf, zero, naive_patch, film_patch], loc=None)
        else:
            ax.legend(handles=[truth, inf, naive_patch, film_patch], loc="lower right")

        ax.set_xticks((x1 + x2) / 2)
        ax.set_xticklabels(nums)
        ax.set_xlabel("Num Real Tasks")

    axs[0].set_ylabel("$\\log \\mathcal{L}$")
    if name is not None:
        save(name)
    plt.show()


def heatmap(e, name=None):
    diffs = np.zeros((len(ls), len(nums_tasks)))
    for i, l in enumerate(tqdm(ls)):
        for j, num_tasks in enumerate(tqdm(nums_tasks)):
            film = e.at(l=l, num=num_tasks, tuner=TunerType.film)["val_lik"].mean()
            naive = e.at(l=l, num=num_tasks, tuner=TunerType.naive)["val_lik"].mean()
            diffs[i, j] = film - naive

    sns.heatmap(
        diffs,
        xticklabels=nums_tasks,
        yticklabels=ls,
        cmap="seismic",
        vmin=-0.075,
        vmax=0.075,
        annot=True,
    )

    if name is not None:
        save(name)
    plt.show()


def heatmap_noise(e, name=None):
    diffs = np.zeros((len(noises), len(nums_tasks)))
    for i, noise in enumerate(tqdm(noises)):
        for j, num_tasks in enumerate(tqdm(nums_tasks)):
            if num_tasks == "inf":
                num_tasks = np.inf
            film = e.at(noise=noise, num=num_tasks, tuner=TunerType.film)[
                "val_lik"
            ].mean()
            naive = e.at(noise=noise, num=num_tasks, tuner=TunerType.naive)[
                "val_lik"
            ].mean()
            diffs[i, j] = naive - film

    sns.heatmap(
        diffs,
        xticklabels=nums_tasks,
        yticklabels=noises,
        cmap="seismic",
        vmin=-0.03,
        vmax=0.03,
        annot=True,
    )

    if name is not None:
        save(name)
    plt.show()


# heatmap_noise(e)

gap_plots(e)
# heatmap(e, "heatmap_multiscale")
# %%
e.df[(e.df["num_tasks"] == np.inf) & (e.df["seed"] == 10) & (e.df["noise"] == 0.0125)]
# %%
e.at(l=0.05)

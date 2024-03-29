import os
import lab as B
import torch

from runspec import Sim2RealSpec, SimRunSpec
from config import sim2real_spec


def ensure_exists(path):
    os.makedirs(path, exist_ok=True)


def with_err(vals, err=None, and_lower=False, and_upper=False):
    """Print the mean value of a list of values with error."""
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    if err is None:
        err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    res = f"{mean:10.5f} +- {err:10.5f}"
    if and_lower:
        res += f" ({mean - err:10.5f})"
    if and_upper:
        res += f" ({mean + err:10.5f})"
    return res


def save_model(model, objective_val, epoch, spec, path):
    torch.save(
        {
            "weights": model.state_dict(),
            "objective": objective_val,
            "epoch": epoch,
            "spec": spec,
        },
        path,
    )


def load_weights(model, path, lik_only=False):
    state = torch.load(path, map_location=sim2real_spec.device)
    val_lik = state["objective"]
    if lik_only:
        return None, val_lik

    weights = state["weights"]
    model.load_state_dict(weights)
    return model, val_lik


def get_exp_dir_base(model, arch, sim_l, noise):
    l_str = float_str(sim_l)
    return f"./outputs/{model}_{arch}/l_sim_{l_str}/noise_{noise:.3g}"


def get_exp_dir_sim(s: SimRunSpec):
    base = get_exp_dir_base(
        s.model.model, s.model.arch, s.data.lengthscale, s.data.noise
    )
    return f"{base}/sim"


def float_str(x):
    if isinstance(x, float):
        return f"{x:.3g}"
    elif isinstance(x, tuple):
        return "-".join([float_str(el) for el in x])


def get_exp_dir_sim2real(s: Sim2RealSpec):
    base = get_exp_dir_base(s.model.model, s.model.arch, s.sim.lengthscale, s.sim.noise)
    num_tasks = s.real.num_tasks_train if not s.real.inf_tasks else "inf"

    l_str = float_str(s.real.lengthscale)
    n_str = float_str(s.real.noise)
    dataspec = f"l_{l_str}_noise_{n_str}_{num_tasks}_tasks"
    tune_dir = f"{base}/tuned/{dataspec}/{s.tuner}/seed_{s.real.train_seed}"
    sim_dir = f"{base}/sim"
    return sim_dir, tune_dir


def get_exp_dir(config, l_real=None, num_tasks_real=None, tuner_type=None):
    dim_x = config["dim_x"]
    dim_y = config["dim_y"]
    model_str = config["model"]
    arch = config["arch"]
    l_pretrained = config["lengthscale_sim"]
    noise = config["noise"]

    exp_dir = f"./outputs/x_{dim_x}_y_{dim_y}/{model_str}_{arch}/l_sim_{l_pretrained:.3g}/noise_{noise:.3g}"
    if l_real is None or num_tasks_real is None or tuner_type is None:
        # We are dealing with a pretrained "sim only" model.
        return f"{exp_dir}/sim"
    else:
        return f"{exp_dir}/tuned/l_real_{l_real:.3g}/num_real_tasks_{num_tasks_real}/{tuner_type}"


# These are functions to ensure consistency across files.
def get_train_plot_dir(exp_dir):
    return f"{exp_dir}/train_plots"


def get_model_dir(exp_dir):
    return f"{exp_dir}/models"


def get_best_model_path(model_dir):
    return f"{model_dir}/best.torch"


def get_latest_model_path(model_dir):
    return f"{model_dir}/latest.torch"


def get_paths(exp_dir):
    model_dir = get_model_dir(exp_dir)
    train_plot_dir = get_train_plot_dir(exp_dir)
    best_model_path = get_best_model_path(model_dir)
    latest_model_path = get_latest_model_path(model_dir)
    return train_plot_dir, best_model_path, latest_model_path, model_dir


def runspec_sim2real(config, real_lengthscale, real_num_tasks_train, tuner):
    keys = [
        "dim_embedding",
        "enc_same",
        "num_layers",
        "residual",  # Use residual connections?
        "affine",  # Use FiLM layers?
        "freeze_affine",  # Freeze affine layers during (pre) training?
        "kernel_size",  # Handled by receptive field
        "conv_receptive_field",
        "conv_channels",
        "margin",
        "mean_diff",
        "transform",
        "normalise_obj",
        "unet_resize_convs",
        "unet_resize_conv_interp_method",
        "unet_channels",
        "unet_strides",
        "num_basis_functions",
        "encoder_scales_learnable",
        "num_epochs",
        "batch_size",
        "rate",
        "tune_rate",
        "num_samples",
        "sim_num_tasks_train",
        "sim_num_tasks_val",
        "epoch_size",
        "real_num_tasks_val",
        "real_inf_tasks",
        "noise",
        "kernel",
        "lengthscale_sim",
        "dim_x",
        "dim_y",
    ]

    result = {}

    for key in keys:
        result[key] = config[key]

    result["real_lengthscale"] = real_lengthscale
    result["real_num_tasks_train"] = real_num_tasks_train
    result["tuner"] = tuner

    return result


def should_eval(epoch, eval_every, epoch_size):
    if epoch == 1:
        return True

    if epoch_size >= eval_every:
        return True

    current_samples_seen = epoch * epoch_size

    if current_samples_seen % eval_every == 0:
        return True
    return False

import os
import lab as B
import torch


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


def save_model(model, objective_val, epoch, path):
    torch.save(
        {
            "weights": model.state_dict(),
            "objective": objective_val,
            "epoch": epoch,
        },
        path,
    )


def load_weights(model, path):
    model.load_state_dict(torch.load(path)["weights"])
    return model


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
    latest_model_path = get_best_model_path(model_dir)
    return train_plot_dir, best_model_path, latest_model_path, model_dir

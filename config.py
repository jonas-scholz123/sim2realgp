import stheno

config = {
    "default": {
        "epochs": None,
        "rate": None,
        "also_ar": False,
    },
    "model": "cnp",
    "epsilon": 1e-8,
    "epsilon_start": 1e-2,
    "cholesky_retry_factor": 1e6,
    "fix_noise": None,
    "fix_noise_epochs": 3,
    "width": 256,
    "dim_embedding": 256,
    "enc_same": False,
    "num_heads": 8,
    "num_layers": 6,
    "unet_channels": (64,) * 6,
    "unet_strides": (1,) + (2,) * 5,
    "conv_channels": 64,
    "encoder_scales": None,
    "fullconvgnp_kernel_factor": 2,
    "mean_diff": None,
    # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
    # doesn't make sense to set it to a value higher of the last hidden layer of
    # the CNN architecture. We therefore set it to 64.
    "num_basis_functions": 64,
    #TODO: what is this?
    "eeg_mode": "random",
    "device": "cpu",
    "normalise_obj": True,
    "num_samples": 20,
    "num_tasks_train": 2**14,
    "num_tasks_val": 2**10,
    "rate": 3e-4,
    "num_epochs": 100,
    "dim_x": 1,
    "dim_y": 1,
    "batch_size": 16,
    "data": "matern",
    "lengthscale_sim": 0.25,
    "kernel": stheno.EQ(),
    "noise": 0.05,
    "output_path": "./outputs",
    "train_path": "/train",
    "sim_model_path": "/sim_trained"
}

dim_x = config["dim_x"]
dim_y = config["dim_y"]
model_str = config["model"]
l_sim = config["lengthscale_sim"]
noise = config["noise"]

exp_dir = f"./outputs/x_{dim_x}_y_{dim_y}/{model_str}/l_sim_{l_sim:.3g}/noise_{noise:.3g}"
config["exp_dir"] = exp_dir

config["train_plot_dir"] = f"{exp_dir}/train_plots"

sim_model_dir = f"{exp_dir}/models/sim"
config["sim_model_dir"] = sim_model_dir
config["sim_best_model_path"] = f"{sim_model_dir}/best.torch"
config["sim_latest_model_path"] = f"{sim_model_dir}/latest.torch"

tuned_model_dir = f"{exp_dir}/models/tuned"
config["tuned_model_dir"] = tuned_model_dir
config["tuned_best_model_path"] = f"{tuned_model_dir}/best.torch"
config["tuned_latest_model_path"] = f"{tuned_model_dir}/latest.torch"
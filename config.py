import stheno
import warnings

from finetuners.get_tuner import TunerType
from matrix.util import ToDenseWarning

warnings.filterwarnings("ignore", category=ToDenseWarning)

config = {
    "default": {
        "epochs": None,
        "rate": None,
        "also_ar": False,
    },
    "wandb": False,
    "model": "convcnp",
    "epsilon": 1e-8,
    "epsilon_start": 1e-2,
    "fix_noise": None,
    "fix_noise_epochs": 3,
    "width": 256,
    "dim_embedding": 256,
    "enc_same": False,
    "num_layers": 8,
    "residual": False,  # Use residual connections?
    "affine": True,  # Use FiLM layers?
    "old": False,
    "kernel_size": None,  # Handled by receptive field
    "unet_channels": (64,) * 6,
    "unet_strides": (1,) + (2,) * 5,
    "unet_resize_convs": True,
    "unet_resize_conv_interp_method": "nearest",
    # To capture all correlations, receptive field should be significantly
    # larger than largest lengthscale. In this case we choose 4 * 0.25 (longest).
    "conv_receptive_field": 1.0,
    "conv_channels": 32,
    "margin": 0.1,
    "mean_diff": None,
    "transform": None,
    # "plot": {
    #    1: {"range": (-1, 1), "axvline": [2]},
    #    2: {"range": ((-1, 1), (-1, 1))},
    # },
    "plot": {
        1: {"range": (-2, 4), "axvline": [2]},
        2: {"range": ((-2, 4), (-2, 4))},
    },
    # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
    # doesn't make sense to set it to a value higher of the last hidden layer of
    # the CNN architecture. We therefore set it to 64.
    "num_basis_functions": 64,
    "arch": "unet",  # unet/conv
    "device": "cpu",
    "normalise_obj": True,
    "num_samples": 20,
    "sim_num_tasks_train": 2**10,
    "sim_num_tasks_val": 2**10,
    # When using different-size real task sets, want one epoch to be consistent
    # e.g. 16 real tasks means epochs are too small to learn.
    "epoch_size": 2**10,
    "real_nums_tasks_train": [2**4, 2**6, 2**8],
    "real_num_tasks_val": 2**8,
    # If False, the same data is re-used each episode. Else, new data is generated
    # and the total number of datapoints is num_epochs * real_num_task_train
    "real_inf_tasks": False,
    "tuners": [TunerType.naive, TunerType.film, TunerType.freeze],
    "rate": 3e-4,
    "tune_rate": 1e-4,
    "num_epochs": 30,
    "dim_x": 1,
    "dim_y": 1,
    "batch_size": 16,
    "lengthscale_sim": 0.25,
    "lengthscales_real": [0.05],
    "kernel": stheno.EQ(),
    "noise": 0.05,
    "encoder_scales_learnable": False,
    "visualise": True,
    "output_path": "./outputs",
    "train_path": "/train",
    "sim_model_path": "/sim_trained",
}

dim_x = config["dim_x"]
dim_y = config["dim_y"]

# Sim lengthscale: 0.25
# Real lengthscales: [0.2, 0.1, 0.05]

# PPU needs to be bigger than the smallest wiggle. For the smallest lengthscale (0.05),
# this corresponds to at least 20.
config["points_per_unit"] = 64
config["encoder_scales"] = 1 / config["points_per_unit"]

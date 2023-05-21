import stheno
import warnings

from finetuners.tuner_types import TunerType
from matrix.util import ToDenseWarning

warnings.filterwarnings("ignore", category=ToDenseWarning)

config = {
    ##############################################
    # Misc.
    ##############################################
    "wandb": True,
    "visualise": True,
    "epsilon": 1e-8,
    "epsilon_start": 1e-2,
    # "plot": {
    #    1: {"range": (-1, 1), "axvline": [2]},
    #    2: {"range": ((-1, 1), (-1, 1))},
    # },
    "plot": {
        1: {"range": (-2, 4), "axvline": [2]},
        2: {"range": ((-2, 4), (-2, 4))},
    },
    "device": "mps",  # cpu/cuda/mps
    ##############################################
    # Model Architecture.
    ##############################################
    "model": "convcnp",
    "arch": "unet",  # unet/conv
    "width": 256,
    "dim_embedding": 256,
    "enc_same": False,
    "num_layers": 8,
    "residual": False,  # Use residual connections?
    "affine": True,  # Use FiLM layers?
    "freeze_affine": True,  # Freeze affine layers during (pre) training?
    "kernel_size": None,  # Handled by receptive field
    # To capture all correlations, receptive field should be significantly
    # larger than largest lengthscale. In this case we choose 4 * 0.25 (longest).
    "conv_receptive_field": 1.0,
    "conv_channels": 32,
    "margin": 0.1,
    "mean_diff": None,
    "transform": None,
    "normalise_obj": True,
    ##############################################
    # UNet Specific Architecture.
    ##############################################
    "unet_resize_convs": True,
    "unet_resize_conv_interp_method": "nearest",
    "unet_channels": (64,) * 5,
    "unet_strides": (1,) + (2,) * 4,
    # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
    # doesn't make sense to set it to a value higher of the last hidden layer of
    # the CNN architecture. We therefore set it to 64.
    "num_basis_functions": 64,
    "encoder_scales_learnable": False,
    # Optimisation
    "tuners": [TunerType.freeze, TunerType.naive, TunerType.film],
    # "tuners": [TunerType.naive],
    "num_epochs": 100,
    "batch_size": 16,
    "rate": 3e-4,
    "tune_rate": 3e-4,
    # Data.
    "num_samples": 20,
    "sim_num_tasks_train": 2**10,
    "sim_num_tasks_val": 2**10,
    # When using different-size real task sets, want one epoch to be consistent
    # e.g. 16 real tasks means epochs are too small to learn.
    "epoch_size": 2**10,
    "real_nums_tasks_train": [2**4, 2**8, 2**10],
    # "real_nums_tasks_train": [2**4],
    "real_num_tasks_val": 2**8,
    # If False, the same data is re-used each episode. Else, new data is generated
    # and the total number of datapoints is num_epochs * real_num_task_train
    "real_inf_tasks": False,
    "noise": 0.05,
    "kernel": stheno.EQ(),
    "lengthscale_sim": 0.25,
    "lengthscales_real": [0.05, 0.1, 0.2],
    # "lengthscales_real": [0.05],
    "dim_x": 1,
    "dim_y": 1,
    # Directories.
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

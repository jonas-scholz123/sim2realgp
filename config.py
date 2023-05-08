import stheno
import warnings
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
    "kernel_size": 33,
    "unet_channels": (64,) * 6,
    # TODO: We should use a stride of 1 in the first layer, but for compatibility
    #    reasons with the models we already trained, we keep it like this.
    "unet_strides": (2,) * 6,
    "conv_receptive_field": 4,
    "margin": 0.1,
    "conv_channels": 64,
    "encoder_scales": None,
    "mean_diff": None,
    "transform": None,
    "plot": {
        1: {"range": (-2, 4), "axvline": [2]},
        2: {"range": ((-2, 2), (-2, 2))},
    },
    # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
    # doesn't make sense to set it to a value higher of the last hidden layer of
    # the CNN architecture. We therefore set it to 64.
    "num_basis_functions": 64,
    "arch": "conv",
    "device": "cpu",
    "normalise_obj": True,
    "num_samples": 20,
    "sim_num_tasks_train": 2**10,
    "sim_num_tasks_val": 2**10,
    "real_num_tasks_train": 2**12,
    "real_num_tasks_val": 2**10,
    "rate": 3e-4,
    "num_epochs": 50,
    "dim_x": 1,
    "dim_y": 1,
    "batch_size": 16,
    "lengthscale_sim": 0.25,
    "kernel": stheno.EQ(),
    "noise": 0.00,
    "output_path": "./outputs",
    "train_path": "/train",
    "sim_model_path": "/sim_trained",
}

dim_x = config["dim_x"]
dim_y = config["dim_y"]
if dim_x == 1:
    config["points_per_unit"] = 64
elif dim_x == 2:
    # Reduce the PPU to reduce memory consumption.
    config["points_per_unit"] = 32
    # Since the PPU is reduced, we can also take off a layer of the UNet.
    config["unet_strides"] = config["unet_strides"][:-1]
    config["unet_channels"] = config["unet_channels"][:-1]

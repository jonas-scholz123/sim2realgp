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
    "device": "mps",
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
    "noise": 0.00,
    "output_path": "./outputs",
    "train_path": "/train",
    "sim_model_path": "/sim_trained"
}
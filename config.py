import stheno
import warnings

from finetuners.tuner_types import TunerType
from matrix.util import ToDenseWarning

from dataclasses import replace

from runspec import (
    SimRunSpec,
    ModelSpec,
    DataSpec,
    ConvSpec,
    UNetSpec,
    OptSpec,
    Directories,
    OutputSpec,
    Sim2RealSpec,
)

warnings.filterwarnings("ignore", category=ToDenseWarning)

config = {
    "tuners": [TunerType.film, TunerType.naive],
    # "tuners": [TunerType.filmlinear],
    "real_nums_tasks_train": [2**10],
    # "real_nums_tasks_train": [2**8, 2**10],
    "lengthscales_real": [0.1, 0.2],
    "noises_real": [0.0125, 0.025, 0.1, 0.2],
    "seeds": list(range(10, 15)),
}

out = OutputSpec(
    wandb=True,
    visualise=True,
    plot={
        1: {"range": (-2, 4), "axvline": [2]},
    },
    eval_every=2**4,
    ignore_previous=True,
)

model = ModelSpec(
    model="convcnp",
    arch="unet",
    width=256,
    dim_embedding=256,
    num_layers=8,
    num_basis_functions=64,
    kernel_size=None,
    residual=False,
    affine=True,
    freeze_affine=True,
    encoder_scales=1 / 64,
    encoder_scales_learnable=False,
    normalise_obj=True,
    margin=0.1,
    mean_diff=None,
    transform=None,
    conv=ConvSpec(
        receptive_field=1.0,
        channels=32,
    ),
    unet=UNetSpec(
        resize_convs=True,
        resize_conv_interp_method="nearest",
        channels=(64,) * 5,
        strides=(1,) + (2,) * 4,
    ),
)

model_mini = ModelSpec(
    model="convcnp_mini",
    arch="unet",
    width=256,
    dim_embedding=256,
    num_layers=6,
    num_basis_functions=64,
    kernel_size=None,
    residual=False,
    affine=True,
    freeze_affine=True,
    encoder_scales=1 / 64,
    encoder_scales_learnable=False,
    normalise_obj=True,
    margin=0.1,
    mean_diff=None,
    transform=None,
    conv=ConvSpec(
        receptive_field=1.0,
        channels=32,
    ),
    unet=UNetSpec(
        resize_convs=True,
        resize_conv_interp_method="nearest",
        channels=(16,) * 3,
        strides=(1,) + (2,) * 2,
    ),
)

model_xl = ModelSpec(
    model="convcnp_xl",
    arch="unet",
    width=256,
    dim_embedding=256,
    num_layers=6,
    num_basis_functions=64,
    kernel_size=None,
    residual=False,
    affine=True,
    freeze_affine=True,
    encoder_scales=1 / 64,
    encoder_scales_learnable=False,
    normalise_obj=True,
    margin=0.1,
    mean_diff=None,
    transform=None,
    conv=ConvSpec(
        receptive_field=1.0,
        channels=32,
    ),
    unet=UNetSpec(
        resize_convs=True,
        resize_conv_interp_method="nearest",
        channels=(128,) * 9,
        strides=(1,) + (2,) * 8,
    ),
)


dirs = Directories(
    output_path="./outputs",
    train_path="/train",
    sim_model_path="/sim_trained",
)

opt = OptSpec(
    num_epochs=500,
    batch_size=16,
    lr=3e-4,
)


sim_data = DataSpec(
    num_samples=20,
    num_tasks_train=2**10,
    num_tasks_val=2**10,
    inf_tasks=True,
    noise=0.05,
    kernel=stheno.EQ(),
    lengthscale=0.25,
    ppu=64,
    dim_x=1,
    dim_y=1,
    train_seed=10,
)


real_data = replace(
    sim_data,
    inf_tasks=False,
    num_tasks_val=2**6,
)

sim_spec = SimRunSpec(
    device="cuda",
    out=out,
    data=sim_data,
    model=model,
    opt=opt,
    dirs=dirs,
)

sim2real_spec = Sim2RealSpec(
    device="cuda",
    tuner=None,
    out=out,
    sim=sim_data,
    real=real_data,
    model=model,
    opt=opt,
    dirs=dirs,
)

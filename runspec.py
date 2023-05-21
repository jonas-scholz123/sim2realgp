from dataclasses import dataclass
import stheno


@dataclass
class DataSpec:
    num_samples: int
    num_tasks_train: int
    num_tasks_val: int
    inf_tasks: bool
    noise: float
    kernel: stheno.kernel
    lengthscale: float
    ppu: int
    dim_x: int
    dim_y: int


@dataclass
class ConvSpec:
    receptive_field: int
    channels: int


@dataclass
class UNetSpec:
    resize_convs: bool
    resize_conv_interp_method: str
    channels: tuple
    strides: tuple


@dataclass
class ModelSpec:
    model: str
    arch: str
    width: int
    dim_embedding: int
    num_layers: int
    num_basis_functions: int
    kernel_size: int
    residual: bool
    affine: bool
    freeze_affine: bool
    encoder_scales: float
    encoder_scales_learnable: bool
    normalise_obj: bool
    margin: float
    mean_diff: float
    transform: any
    conv: ConvSpec
    unet: UNetSpec


@dataclass
class OptSpec:
    num_epochs: int
    batch_size: int
    epoch_size: int
    lr: float


@dataclass
class Directories:
    output_path: str
    train_path: str
    sim_model_path: str


@dataclass
class OutputSpec:
    wandb: bool
    visualise: bool
    plot: dict


@dataclass
class SimRunSpec:
    """
    Class that specifies all configurable items of a simulation run.
    """

    device: str
    out: OutputSpec
    data: DataSpec
    model: ModelSpec
    opt: OptSpec
    dirs: Directories
    stage: str = "sim"


spec = SimRunSpec(
    device="mps",
    out=OutputSpec(
        wandb=False,
        visualise=True,
        plot={
            1: {"range": (-2, 4), "axvline": [2]},
            2: {"range": ((-2, 4), (-2, 4))},
        },
    ),
    data=DataSpec(
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
    ),
    model=ModelSpec(
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
    ),
    opt=OptSpec(
        num_epochs=100,
        batch_size=16,
        epoch_size=2**10,
        lr=3e-4,
    ),
    dirs=Directories(
        output_path="./outputs",
        train_path="/train",
        sim_model_path="/sim_trained",
    ),
)

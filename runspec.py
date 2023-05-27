from dataclasses import dataclass
import stheno

from finetuners.tuner_types import TunerType


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
    eval_every: int


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


@dataclass
class Sim2RealSpec:
    """
    Class that specifies all configurable items of a sim2real run.
    """

    device: str
    tuner: TunerType
    out: OutputSpec
    sim: DataSpec
    real: DataSpec
    model: ModelSpec
    opt: OptSpec
    dirs: Directories
    stage: str = "tune"

from enum import Enum
from finetuners.film_tuner import FilmTuner
from finetuners.naive_tuner import NaiveTuner
from finetuners.freeze_tuner import FreezeTuner
from finetuners.tuner import Tuner
from finetuners.tuner_types import TunerType


def get_tuner(tt: TunerType) -> Tuner:
    if tt == TunerType.naive:
        return NaiveTuner
    if tt == TunerType.film:
        return FilmTuner
    if tt == TunerType.freeze:
        return FreezeTuner

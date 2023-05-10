from enum import Enum
from finetuners.film_tuner import FilmTuner
from finetuners.naive_tuner import NaiveTuner
from finetuners.freeze_tuner import FreezeTuner
from finetuners.tuner import Tuner


class TunerType(Enum):
    naive = 1
    film = 2
    freeze = 3


def get_tuner(tt: TunerType) -> Tuner:
    if tt == TunerType.naive:
        return NaiveTuner
    if tt == TunerType.film:
        return FilmTuner
    if tt == TunerType.freeze:
        return FreezeTuner

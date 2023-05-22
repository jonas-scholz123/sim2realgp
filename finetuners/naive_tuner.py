# %%
from finetuners.tuner import Tuner
import lab as B


class NaiveTuner(Tuner):
    """
    Modify all weights in the model, leaving the model architecture unchanged.
    """

    def __init__(self, initial_model, objective, opt, spec):
        super().__init__(initial_model, objective, opt, spec)

    def modify_model(self, initial_model):
        """
        No modifications made in the NaiveTuner.
        """
        return initial_model

    def name(self):
        return "NaiveTuner"

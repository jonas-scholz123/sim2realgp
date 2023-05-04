#%%
from finetuners.tuner import Tuner
from train import train
import lab as B

class NaiveTuner(Tuner):
    '''
    Modify all weights in the model, leaving the model architecture unchanged.
    '''
    def __init__(self, initial_model, objective, opt, lr):
        super().__init__(initial_model, objective, opt, lr)
    
    def modify_model(self, initial_model):
        '''
        No modifications made in the NaiveTuner.
        '''
        return initial_model
    
    def train_on_batch(self, batch, state):
        state, obj = self.objective(
            state,
            self.model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
        )
        val = -B.mean(obj)
        self.opt.zero_grad(set_to_none=True)
        val.backward()
        self.opt.step()
        return state
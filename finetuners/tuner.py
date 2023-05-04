from abc import abstractmethod, ABC

class Tuner(ABC):
    def __init__(self, initial_model, objective, opt, lr):
        self.model = self.modify_model(initial_model)
        self.objective = objective
        self.opt = opt(self.model.parameters(), lr)
    
    @abstractmethod
    def modify_model(self, initial_model):
        return

    @abstractmethod 
    def train_on_batch(self, batch, state):
        pass
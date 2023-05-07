# %%
import torch
from torch import nn

shape = (10, 10)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(10, 20, 2)
        self.conv2 = nn.Conv1d(20, 20, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        return h


model = Model()
model
# %%
x = torch.rand(shape)
model(x)

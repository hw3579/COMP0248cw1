import netron
import torch 
import train
from train import TotalModel
import os

model = torch.load('results/full_model.pth', weights_only=False)
model = "results/full_model.pth"
netron.start(model) # serve on localhost:9999

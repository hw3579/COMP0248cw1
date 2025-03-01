import netron
import torch 
import train
from train import TotalModel
import os
from train_deeplabv3 import TotalDeepLabV3Plus, Backbone, ASPP, Bottleneck, DeepLabV3PlusDecoder

# Load the model
model = "results/deeplabmodelfull.pth"
netron.start(model) # serve on localhost:9999

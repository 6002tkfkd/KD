# main.py
from lightning.pytorch.cli import LightningCLI
import torch
import model.lit_asc
import data.data_module
import util
import model.backbones

import time
import random
from pytorch_lightning import seed_everything

dynamic_seed = int(time.time()*random.random()) 
seed_everything(dynamic_seed, workers=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

cli = LightningCLI()

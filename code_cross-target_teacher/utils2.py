import pdb
import numpy as np
import os.path as path
import torch
# from torch.utils.serialization import load_lua
from options import opt


def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False
        
        

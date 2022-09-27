import os
from collections import Counter
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch import cuda, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set data directories
traindir = f"data/train"
validdir = f"data/val"
testdir = f"data/test"
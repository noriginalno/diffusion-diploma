import os
import random
import multiprocessing
import gc
import pathlib
import shutil
from functools import reduce
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import pandas as pd
import numpy as np
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.stats import norm, multivariate_normal
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import diffusers
from diffusers import DDPMScheduler
from diffusers.models.embeddings import get_timestep_embedding



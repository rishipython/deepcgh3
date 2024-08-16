import os
import tensorflow as tf             # TensorFlow operations
from flax import linen as nn  # Linen API
import jax
import jax.numpy as jnp  # JAX NumPyimport jax
from jax import lax
import matplotlib.pyplot as plt
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import numpy as np
import optax
import time
import pickle
from skimage.draw import disk
from typing import *
from chex import Array
from einops import rearrange, repeat
import h5py as h5
from tqdm import tqdm
from functools import partial
import cv2
from scipy.spatial import KDTree
import random
from chromatix import Field, ScalarField

a = jnp.ones((10,10))
print(a)
print(a.devices())
print("Yay!")

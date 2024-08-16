import os
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
from chromatix.ops import quantize
from chromatix.functional import wrap_phase, compute_transfer_propagator, ff_lens, transfer_propagate
from chromatix.functional.sources import plane_wave
from copy import deepcopy
import fouriernet.jax as fouriernet
import re
from math import prod

def serialize(save_path: str, fname: str, obj: Any):
    with open(os.path.join(save_path, fname), 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def deserialize(save_path: str, fname: str) -> Any:
    with open(os.path.join(save_path, fname), 'rb') as f:
        return pickle.load(f)

batch_size = 8

def cosine3D(y_true, y_pred):
    intersect = jnp.sum(y_true*y_pred, axis=(1, 2, 3))
    fsum = jnp.sum(y_true**2, axis=(1, 2, 3))
    ssum = jnp.sum(y_pred**2, axis=(1, 2, 3))
    cosine = 1 - jnp.mean((intersect / (fsum * ssum)))
    return cosine

def loss_cosine(predictions, targets):
    # predictions = jnp.clip(predictions, a_min=0)
    # predictions = predictions / predictions.max()

    return optax.cosine_distance(predictions = predictions.reshape(predictions.shape[0], -1),
                                 targets = targets.reshape(targets.shape[0], -1).astype(jnp.float32),
                                 epsilon=1e-6).mean()

def loss_acc(predictions, targets):
    # predictions = jnp.clip(predictions, a_min=0)
    # predictions = predictions / predictions.max()
    
    denominator = jnp.sqrt(jnp.sum(predictions**2, axis=(1, 2, 3)) * jnp.sum(targets**2, axis=(1, 2, 3)))
    loss = 1.0 - jnp.mean((jnp.sum(predictions * targets, axis=(1, 2, 3)) + 1) / (denominator + 1), axis=0)
    return loss

num_devices = 1 #int(sys.argv[1])
size = 512 #int(sys.argv[2])
key = repeat(jax.random.PRNGKey(4), "k -> d k", d=num_devices)
rng = np.random.default_rng(4)
folder_name = os.getcwd().split('/')[-1]
save_path = f'/nrs/turaga/athavaler/experiments/{folder_name}'
os.makedirs(save_path, exist_ok=True)

z_planes = np.array([0.0, 0.05, 0.10])

class DeepCGH(nn.Module):
    shape: Tuple[int, int] = (512, 512)
    spacing: float = 9.2e-6
    # z: Array = 0.05 * np.array([-1, 0, 1])
    f: float = 0.2
    n: float = 1.0
    NA: Optional[float] = None
    N_pad: int = 0
    spectrum: Array = 1.04e-6
    spectral_density: Array = 1.0
    bit_depth: int = 8
    interleave_factor: int = 16

    @nn.compact
    def __call__(self, target: Array, mode: str = "train") -> Field:
        propagator = self.variable(
            "state",
            "kernel",
            lambda: jnp.fft.fftshift(
                compute_transfer_propagator(
                    plane_wave(self.shape, self.spacing, self.spectrum, self.spectral_density),
                    z_planes,
                    self.n,
                ),
                axes=(-4, -3)
            ),
        ).value
        phase = self.target_to_phase(target, mode=mode)
        field = ScalarField.create(
            self.spacing,
            self.spectrum,
            self.spectral_density,
            rearrange(jnp.exp(1j * phase), "b h w -> b 1 h w 1 1")
        )
        field *= propagator[jnp.newaxis, ...]
        field = ff_lens(field, self.f, self.n)
        return field, phase

    def target_to_phase(self, target: Array, mode: str = "train") -> Array:
        def interleave(x: Array) -> Array:
            return rearrange(x, "b (h hs) (w ws) c -> b h w (c hs ws)", hs=self.interleave_factor, ws=self.interleave_factor)

        def deinterleave(x: Array) -> Array:
            return rearrange(x, "b h w (c hs ws) -> b (h hs) (w ws) c", hs=self.interleave_factor, ws=self.interleave_factor)

        def upsample(x: Array) -> Array:
            return repeat(x, "b h w c -> b (h hs) (w ws) c", hs=2, ws=2)

        def cbn(x: Array, features: int) -> Array:
            x = nn.Conv(features, (3, 3), padding="SAME", kernel_init=nn.initializers.glorot_uniform())(x)
            x = nn.relu(x)
            x = nn.BatchNorm(use_running_average=(mode != "train"))(x)
            x = nn.Conv(features, (3, 3), padding="SAME", kernel_init=nn.initializers.glorot_uniform())(x)
            x = nn.relu(x)
            x = nn.BatchNorm(use_running_average=(mode != "train"))(x)
            return x

        def cc(x: Array, features: int) -> Array:
            x = nn.Conv(features, (3, 3), padding="SAME", kernel_init=nn.initializers.glorot_uniform())(x)
            x = nn.relu(x)
            x = nn.Conv(features, (3, 3), padding="SAME", kernel_init=nn.initializers.glorot_uniform())(x)
            x = nn.relu(x)
            return x

        x0 = interleave(target)
        x1 = cbn(x0, 64)
        x2 = nn.max_pool(x1, (2, 2), strides=(2, 2), padding="SAME")
        x2 = cbn(x2, 128)
        encoded = nn.max_pool(x2, (2, 2), strides=(2, 2), padding="SAME")
        encoded = cc(encoded, 256)
        x3 = upsample(encoded)
        x3 = jnp.concatenate([x3, x2], axis=-1)
        x3 = cc(x3, 128)
        x4 = upsample(x3)
        x4 = jnp.concatenate([x4, x1], axis=-1)
        x4 = cc(x4, 64)
        x4 = cc(x4, 128)
        x4 = jnp.concatenate([x4, x0], axis=-1)
        phase = nn.Conv(self.interleave_factor**2, (3, 3), padding="SAME", kernel_init=nn.initializers.glorot_uniform())(x4)
        phase = deinterleave(phase)
        amplitude = nn.Conv(self.interleave_factor**2, (3, 3), padding="SAME", kernel_init=nn.initializers.glorot_uniform())(x4)
        amplitude = nn.relu(amplitude)
        amplitude = deinterleave(amplitude)
        field = amplitude * jnp.exp(1j * phase)
        phase = jnp.angle(jnp.fft.ifft2(jnp.fft.ifftshift(field, axes=(1, 2)), axes=(1, 2)))
        phase = quantize(wrap_phase(phase), self.bit_depth, range=(-jnp.pi, jnp.pi))
        phase = jnp.squeeze(phase, axis=-1)
        return phase

batch_size = 16
model = DeepCGH(shape=(size, size), interleave_factor=size // 32)
variables = jax.pmap(model.init)(
    repeat(jax.random.PRNGKey(3331), "k -> d k", d=num_devices),
    jnp.zeros((num_devices, batch_size // num_devices, size, size, 1))
)
print(jax.tree_util.tree_map(lambda x: x.shape, variables))
params = deserialize(save_path, "params.pkl")
state = variables["state"]
batch_stats = deserialize(save_path, "batch_stats.pkl")
del variables
optimizer = optax.adam(learning_rate=1e-4)
opt_state = jax.pmap(optimizer.init)(params)

test_data = []
for i in range(100):
    print(i)
    batch = deserialize('', f'/nrs/turaga/athavaler/experiments/exp065_check_z_thing/sweep_000_lr=1e-5_opt=lion_embedsize=2048_fixedtrainingset=T_fixedrandi=T/cache_test_{i}.pkl')['batch']
    for j in range(8):
        test_data.append([batch_ele[j][None,None,...] for batch_ele in batch])

times = []
losses = []

@partial(jax.pmap, axis_name="devices")
def run_model(batch, params, state, batch_stats):
    approx, phase = model.apply({"params": params, "state": state, "batch_stats": batch_stats},jnp.moveaxis(batch[-1], 1, -1), mode="test")
    approx = rearrange(approx.intensity, "b z y x 1 1 -> b z y x")
    return approx, phase

for i, batch in enumerate(test_data):
  t0 = time.time()
  recon, slm_phase = run_model(batch, params, state, batch_stats)
  recon = recon[0]
  slm_phase = slm_phase[0]

  serialize(save_path, f'recon_{i}.pkl', recon)
  serialize(save_path, f'slm_{i}.pkl', slm_phase)
  serialize(save_path, f'batch_{i}.pkl', batch)

  time_passed = time.time() - t0
  times.append(time_passed)

  losses.append({
        'loss_cosine': loss_cosine(recon, batch[3]),
        'loss_acc': loss_acc(recon, batch[3])
    })
  print(f"test batch: {i}/{len(test_data)}, "
        f"time: {time_passed}, "
        f"loss cosine: {losses[-1]['loss_cosine']}, "
        f"loss acc: {losses[-1]['loss_acc']}, "
    )

print("mean time", np.mean(np.array(times)))
print("mean loss cosine", np.mean(np.array([loss_item['loss_cosine'] for loss_item in losses])))
print("mean loss acc", np.mean(np.array([loss_item['loss_acc'] for loss_item in losses])))
    
# datetime = time.strftime("%Y%m%d-%H%M%S")

serialize(save_path, f'times.pkl', times)
print(f'times.pkl')

serialize(save_path, f'losses.pkl', losses)
print(f'losses.pkl')
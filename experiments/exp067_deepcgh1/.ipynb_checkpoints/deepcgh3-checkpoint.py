import jax
from jax.experimental.host_callback import id_print
import jax.numpy as jnp
import numpy as np
import optax
from optax import adam
from flax import linen as nn
from flax.core import frozen_dict
from chex import Array, ArrayTree

import pickle
import os, sys, datetime
from functools import partial
from typing import *
from einops import rearrange, repeat

# from fouriernet.jax import *
from flax.serialization import to_state_dict, from_state_dict
from chromatix import Field, ScalarField
from chromatix.elements.utils import trainable
from chromatix.systems import OpticalSystem
from chromatix.elements import PlaneWave, AmplitudeMask, PhaseMask, FFLens, Propagate, trainable
from chromatix.utils.initializers import *
from chromatix.utils.data import siemens_star
from chromatix.functional.propagation import transfer_propagate, compute_transfer_propagator
from chromatix.functional.phase_masks import wrap_phase
from chromatix.functional.sources import plane_wave
from chromatix.functional.lenses import ff_lens
from chromatix.ops import quantize
# from holoscope.optics import *
# from holoscope.optimization import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass, field
import zarr

num_devices = 1 #int(sys.argv[1])
size = 512 #int(sys.argv[2])
key = repeat(jax.random.PRNGKey(4), "k -> d k", d=num_devices)
rng = np.random.default_rng(4)
# save_path = f"/nrs/turaga/debd/lfm/holoscope-experiments/{os.getcwd().split('/')[-3]}/{os.getcwd().split('/')[-2]}/{os.getcwd().split('/')[-1]}"
folder_name = os.getcwd().split('/')[-1]
save_path = f'/nrs/turaga/athavaler/experiments/{folder_name}'
os.makedirs(save_path, exist_ok=True)


class DeepCGH(nn.Module):
    shape: Tuple[int, int] = (512, 512)
    spacing: float = 9.2e-6
    z: Array = field(default_factory=lambda: 0.05 * np.array([0]))
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
            lambda: compute_transfer_propagator(
                plane_wave(self.shape, self.spacing, self.spectrum, self.spectral_density),
                self.z,
                self.n,
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
        return field

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
params = variables["params"]
state = variables["state"]
batch_stats = variables["batch_stats"]
del variables
optimizer = adam(learning_rate=1e-4)
opt_state = jax.pmap(optimizer.init)(params)


def serialize(save_path: str, fname: str, obj: Any):
    with open(os.path.join(save_path, fname), 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def deserialize(save_path: str, fname: str) -> Any:
    with open(os.path.join(save_path, fname), 'rb') as f:
        return pickle.load(f)


def loss_fn(params: ArrayTree, state: ArrayTree, batch_stats: ArrayTree, target: Array) -> Tuple[Array, ArrayTree]:
    approx, updates = model.apply({"params": params, "state": state, "batch_stats": batch_stats}, target, mutable="batch_stats")
    approx = rearrange(approx.intensity, "b z y x 1 1 -> b y x z")
    denominator = jnp.sqrt(jnp.sum(approx**2, axis=(1, 2, 3)) * jnp.sum(target**2, axis=(1, 2, 3)))
    loss = 1 - jnp.mean((jnp.sum(approx * target, axis=(1, 2, 3)) + 1) / (denominator + 1), axis=0)
    return loss, updates["batch_stats"]


@partial(jax.pmap, axis_name="devices")
def step(params: ArrayTree, state: ArrayTree, batch_stats: ArrayTree, target: Array, opt_state: ArrayTree) -> Tuple[ArrayTree, ArrayTree, ArrayTree, float]:
    (loss, batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, batch_stats, target)
    # grads = grads.unfreeze()
    grads = jax.lax.pmean(grads, axis_name="devices")
    # grads = frozen_dict.freeze(grads)
    # batch_stats = batch_stats.unfreeze()
    batch_stats = jax.lax.pmean(batch_stats, axis_name="devices")
    # batch_stats = frozen_dict.freeze(batch_stats)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, batch_stats, opt_state, loss


dataset = zarr.load(f"/nrs/turaga/athavaler/experiments/{os.getcwd().split('/')[-1]}/_SHP({size}, {size}, 1)_N2560_SZ5_INT1_Crowd[5, 6]_CNTFalse/train.zarr")
max_iterations = 128 * 10
shuffle = 128
losses = np.zeros((max_iterations))
step_times = np.zeros((max_iterations))
indices = np.arange(dataset.shape[0])
batched_indices = np.array_split(indices, 128)
devices = jax.devices("gpu")
for iteration in range(max_iterations):
    if iteration % shuffle == 0:
        indices = np.random.permutation(dataset.shape[0])
        batched_indices = np.array_split(indices, 128)
    target = dataset[batched_indices[iteration % len(batched_indices)]]
    target = np.split(target, num_devices)
    target = jax.device_put_sharded(target, devices)
    target.block_until_ready()
    _start = datetime.datetime.now()
    params, batch_stats, opt_state, loss = step(params, state, batch_stats, target, opt_state)
    jax.tree_util.tree_leaves(params)[0].block_until_ready()
    step_time = (datetime.datetime.now() - _start).total_seconds()
    step_times[iteration] = step_time
    losses[iteration] = np.mean(np.array(loss))
    print(datetime.datetime.now(), iteration, loss, step_time)

assert jnp.all(jax.tree_util.tree_leaves(params)[0][0] == jax.tree_util.tree_leaves(params)[0][-1]), "Parameters different on different GPUs"
assert jnp.all(jax.tree_util.tree_leaves(batch_stats)[0][0] == jax.tree_util.tree_leaves(batch_stats)[0][-1]), "Batch stats different on different GPUs"
serialize(save_path, "params_2.pkl", params)
serialize(save_path, "batch_stats_2.pkl", batch_stats)
np.save(f"{save_path}/losses_2", losses)
np.save(f"{save_path}/step_times_2", step_times)
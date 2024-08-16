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

from fouriernet.jax import *
from flax.serialization import to_state_dict, from_state_dict
from chromatix import Field
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
from holoscope.optics import *
from holoscope.optimization import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

num_devices = 1 #int(sys.argv[1])
size = 128 #int(sys.argv[2])
key = repeat(jax.random.PRNGKey(4), "k -> d k", d=num_devices)
rng = np.random.default_rng(4)
# save_path = f"/nrs/turaga/debd/lfm/holoscope-experiments/{os.getcwd().split('/')[-3]}/{os.getcwd().split('/')[-2]}/{os.getcwd().split('/')[-1]}"
save_path = '/nrs/turaga/athavaler/experiments/exp008_deepcgh1'

os.makedirs(save_path, exist_ok=True)


class DeepCGH(nn.Module):
    shape: Tuple[int, int] = (512, 512)
    spacing: float = 9.2e-6
    z: Array = 0.05 * np.array([-1, 0, 1])
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
    jnp.zeros((num_devices, batch_size // num_devices, size, size, 3))
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
    grads = grads.unfreeze()
    grads = jax.lax.pmean(grads, axis_name="devices")
    grads = frozen_dict.freeze(grads)
    batch_stats = batch_stats.unfreeze()
    batch_stats = jax.lax.pmean(batch_stats, axis_name="devices")
    batch_stats = frozen_dict.freeze(batch_stats)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, batch_stats, opt_state, loss


# dataset = zarr.load(f"/nrs/turaga/debd/lfm/holoscope-experiments/010_chromatix_paper_experiments/17_generate_deep_cgh_data/_SHP({size}, {size}, 3)_N2560_SZ10_INT[0.1, 1]_Crowd[27, 48]_CNTFalse/train.zarr")

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
from chromatix.functional import wrap_phase, compute_transfer_propagator, ff_lens

def draw_disks(image_size, coordinates, radii, disk_weights):
    image = np.zeros(image_size, dtype=np.uint8)

    cv2.circle(img=image,
                center=coordinates,
                radius=radii,
                color=disk_weights,
                thickness=-1)

    return image


def draw_disksW(image_size, coordinates, radii, disk_intensities, pixel_weights, background_weight):
    image = np.zeros(image_size, dtype=np.uint8)
    weight_image = np.ones(image_size, dtype=np.float32) * background_weight

    for ind, coord, r, di in zip(range(len(coordinates)), coordinates, radii, disk_intensities):
        cv2.circle(img=image,
                    center=(coord[0], coord[1]),
                    radius=r,
                    color=int(di),
                    thickness=-1)
        if pixel_weights is not None:
            cv2.circle(img=weight_image,
                        center=(coord[0], coord[1]),
                        radius=r,
                        color=float(pixel_weights[ind]),
                        thickness=-1)

    return image, weight_image

def normalize_2_uin8(y):
    x = y.astype(np.float32)
    x -= np.min(x, axis = [1, 2], keepdims=True)
    x /= np.max(x, axis = [1, 2], keepdims=True)
    x *= 255
    return np.round(x).astype(np.uint8)


class RandoDotGenerator:
    def __init__(self,
                 N,
                 n_points,
                 radius,
                 intensity,
                 shape,
                 z_range,
                 z_noise,
                 z_quantization = 20,
                 background_weight = 0.1,
                 output_arguments = None):
        assert len(shape) == 3, 'Shape must specify three dimensions, shape parameter is: {}'.format(len(shape))
        self.N = N
        self.shape = shape
        self.z_range = z_range
        self.z_noise = z_noise
        self.quantization = z_quantization
        self.n_points = n_points
        self.num_planes = shape[0]
        # Check if data is 3D
        assert self.num_planes == 1 or (isinstance(self.z_range, list) and len(self.z_range) == 2), "For 3D images a list for `z_range` has to be provided."

        self.radius = radius
        self.disk_intensity = intensity
        self.background_weight = background_weight

        self.count = 0

        if output_arguments is None:
            vector_arguments = {
                'x': True,
                'y': True,
                'z': True if self.num_planes != 1 else False,
                'radius': True,
                'intensity': True}
            image_arguments = {
                'image': True,
                'pixel_weights': True}
        else:
            vector_arguments = {
                'x': output_arguments['x'],
                'y': output_arguments['y'],
                'z': False if self.num_planes == 1 else output_arguments['z'], # dummy proofing this
                'radius': output_arguments['radius'],
                'intensity': output_arguments['intensity']}
            image_arguments = {
                'image': output_arguments['image'],
                'pixel_weights': output_arguments['pixel_weights']}

        self.vector_arguments = vector_arguments
        self.image_arguments = image_arguments
        self.__get_randomized_coords()


    def no_overlap(self, new_coord, existing_coords, new_radius, existing_radii):
        if len(existing_coords) == 0:
            return True
        distances = np.sqrt(np.sum((existing_coords - new_coord) ** 2, axis=1))
        min_distance = new_radius + existing_radii
        return np.all(distances >= min_distance)


    def get_pixel_weights(self):
        disk_area = np.pi * (self.radii ** 2)

        if isinstance(self.disk_intensity, list):
            min_area, max_area = np.min(disk_area), np.max(disk_area)
            normalized_area = (disk_area - min_area) / (max_area - min_area)
        else:
            normalized_area = disk_area

        if isinstance(self.disk_intensity, list):
            min_intensity, max_intensity = min(self.disk_intensity), max(self.disk_intensity)
            normalized_intensities = (self.disk_intensities - min_intensity) / (max_intensity - min_intensity)
        else:
            normalized_intensities = self.disk_intensities

        epsilon = 1e-6
        self.pixel_weights = 1 / (normalized_area * normalized_intensities + epsilon)
        self.pixel_weights /= np.max(self.pixel_weights)
        self.pixel_weights += 1

        # self.background_weights = np.sum(self.pixel_weights, axis = -1) * self.background_weight


    def __get_randomized_coords(self):
        if isinstance(self.radius, list):
            self.radii = np.random.randint(self.radius[0], self.radius[1], size = (self.N, self.n_points))
        else:
            self.radii = np.ones((self.N, self.n_points), dtype=np.uint8) * self.radius

        if isinstance(self.disk_intensity, list):
            self.disk_intensities = np.random.randint(self.disk_intensity[0], self.disk_intensity[1], size=(self.N, self.n_points))
        else:
            self.disk_intensities = np.ones((self.N, self.n_points), dtype=np.uint8) * self.disk_intensity

        self.get_pixel_weights()

        self.centery = np.empty((self.N, self.n_points), dtype=np.int32)
        self.centerx = np.empty((self.N, self.n_points), dtype=np.int32)

        if self.num_planes > 1:
            z_basis = np.linspace(self.z_range[0], self.z_range[1], num=self.quantization, endpoint=True)

            z_indices = []
            for _ in range(self.N):
                z_indices.append(random.sample(range(self.quantization), self.num_planes))
            z_indices = np.array(z_indices)

            z_noise = (np.random.rand(self.N, self.num_planes)-0.5) * 2 * self.z_noise * (self.z_range[0]-self.z_range[1])
            self.z_values = z_basis[z_indices] + z_noise
            self.z_values.sort(axis=1)

            self.z = np.zeros_like(self.centerx).astype(np.float32)


        for n in tqdm(range(self.N)):
            existing_coords = [[] for _ in range(self.num_planes)]
            existing_radii = [[] for _ in range(self.num_planes)]

            for p in range(self.n_points):
                max_tries = 100
                num_tries = 0
                while num_tries < max_tries:
                    plane = np.random.randint(low=0, high=self.num_planes) if self.num_planes > 1 else 0
                    y = np.random.randint(low=self.radii[n, p], high=self.shape[1] - self.radii[n, p])
                    x = np.random.randint(low=self.radii[n, p], high=self.shape[2] - self.radii[n, p])
                    if self.no_overlap(np.array([x, y]), np.array(existing_coords[plane]), self.radii[n, p], np.array(existing_radii[plane])):
                        self.centerx[n, p] = x
                        self.centery[n, p] = y
                        if self.num_planes > 1:
                            self.z[n, p] = self.z_values[n, plane]
                        existing_coords[plane].append([x, y])
                        existing_radii[plane].append(self.radii[n, p])
                        break
                    num_tries += 1

                if num_tries == max_tries:
                    break


    def __len__(self):
        return self.N


    def __getitem__(self, idx):
        coords = np.array([self.centerx[idx], self.centery[idx]]).T

        vector_elements = {}
        if self.vector_arguments['x']:
            vector_elements['x'] = self.centerx[idx]

        if self.vector_arguments['y']:
            vector_elements['y'] = self.centery[idx]

        if self.vector_arguments['radius']:
            vector_elements['radius'] = self.radii[idx]

        if self.vector_arguments['intensity']:
            vector_elements['intensity'] = self.disk_intensities[idx]

        if self.num_planes > 1 and (self.image_arguments['image'] or self.image_arguments['pixel_weights']):
            image = np.zeros(self.shape, dtype=np.uint8)
            weight_image = np.zeros(self.shape, dtype=np.float32)
            for i in range(self.num_planes):
                pix_w8 = None if not self.image_arguments['pixel_weights'] else self.pixel_weights[idx, self.z_indices[idx] == i]
                bg_w8 = self.background_weight * (1 if pix_w8 is None else np.min(pix_w8))
                img, wt_img = draw_disksW(self.shape[1:],
                                          coords[self.z[idx] == self.z_values[idx, i], :],
                                          self.radii[idx, self.z[idx] == self.z_values[idx, i]],
                                          self.disk_intensities[idx, self.z[idx] == self.z_values[idx, i]],
                                          pix_w8,
                                          bg_w8)
                image[i, :, :] = img
                weight_image[i, :, :] = wt_img

            if self.vector_arguments['z']:
                vector_elements['z'] = self.z[idx]

        else:
            image = np.zeros(self.shape, dtype=np.uint8)
            weight_image = np.zeros(self.shape, dtype=np.float32)
            pix_w8 = None if not self.image_arguments['pixel_weights'] else self.pixel_weights[idx]
            bg_w8 = self.background_weight * (1 if pix_w8 is None else np.min(pix_w8))
            img, wt_img = draw_disksW(self.shape[1:],
                                              coords,
                                              self.radii[idx],
                                              self.disk_intensities[idx],
                                              pix_w8,
                                              bg_w8)
            image[0, :, :] = img
            weight_image[0, :, :] = wt_img

        output = []

        output.append(np.array([vector_elements[key] for key in self.vector_arguments.keys() if self.vector_arguments[key]]).astype(np.float32).T)

        if self.num_planes > 1:
            output.append(self.z_values[idx])

        if self.image_arguments['image']:
            output.append(image.astype(np.float32))

        if self.image_arguments['pixel_weights']:
            output.append(weight_image)



        return tuple(output)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__()-1:
                self.on_epoch_end()


    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.__get_randomized_coords()

    def get_batch(self, batch_size):
      batch_x = []
      batch_y = []
      for i in range(batch_size):
        data = self.__getitem__(self.count)
        self.count += 1
        if self.count >= self.__len__():
          self.on_epoch_end()
          self.count = 0
        if i >= batch_size: return batch
        batch_x.append(data[0])
        batch_y.append(data[1])
      return (np.array(batch_x), np.array(batch_y))

gen = RandoDotGenerator(N = 100,
        n_points = 5,
        radius = 5,
        intensity = 1,
        shape = (1, 128, 128),
        z_range = 1,
        z_noise = 0.1,
        background_weight = 0.1,
        output_arguments = {'x': True,
                            'y': True,
                            'z': False,
                            'radius': False,
                            'intensity': False,
                            'image': True,
                            'pixel_weights': False})

max_iterations = 128 * 10
shuffle = 128
losses = np.zeros((max_iterations))
step_times = np.zeros((max_iterations))
# indices = np.arange(dataset.shape[0])
# batched_indices = np.array_split(indices, 128)
devices = jax.devices("gpu")
for iteration in range(max_iterations):
    # if iteration % shuffle == 0:
    #     indices = np.random.permutation(dataset.shape[0])
    #     batched_indices = np.array_split(indices, 128)
    # target = dataset[batched_indices[iteration % len(batched_indices)]]
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
serialize(save_path, "params.pkl", params)
serialize(save_path, "batch_stats.pkl", batch_stats)
np.save(f"{save_path}/losses", losses)
np.save(f"{save_path}/step_times", step_times)
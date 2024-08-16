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

save_path = f"/nrs/turaga/athavaler/experiments/{os.getcwd().split('/')[-2]}/{os.getcwd().split('/')[-1]}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

starting_params_path = None
starting_opt_state_path = None

def serialize(save_path: str, fname: str, obj: Any):
    with open(os.path.join(save_path, fname), 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def deserialize(save_path: str, fname: str) -> Any:
    with open(os.path.join(save_path, fname), 'rb') as f:
        return pickle.load(f)

print("Completed imports!")

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
                 output_arguments = None,
                 z_fixed=None):
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
        
        self.z_fixed = z_fixed
        if not isinstance(self.z_fixed, list) and self.z_fixed is not None: self.z_fixed = [self.z_fixed]
        
        assert self.z_fixed is None or len(self.z_fixed) == self.num_planes, "Len of fixed z values needs to be equal to num planes"

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
            if self.z_fixed is not None:
                assert len(self.z_fixed) == self.num_planes, "Len of fixed z values needs to be equal to num planes"
                self.z_values = np.array([self.z_fixed for _ in range(self.N)])
            else:
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
            output.append(self.z[idx])
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
      batch_coords = []
      batch_z = []
      batch_z_values = []
      batch_label = []
      for i in range(batch_size):
        data = self.__getitem__(self.count)
        self.count += 1
        if self.count >= self.__len__():
          self.on_epoch_end()
          self.count = 0
        batch_coords.append(data[0])
        if self.num_planes > 1:
          batch_z.append(data[1])
          batch_z_values.append(data[2])
          batch_label.append(data[3])
        else:
          batch_label.append(data[1])
      return (np.array(batch_coords), np.array(batch_z), np.array(batch_z_values), np.array(batch_label))

class ResBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        out = nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME')(x)
        out = nn.relu(out)
        out = nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME')(out)
        out = nn.relu(out)
        # out = nn.Conv(x.shape[-1], kernel_size=(1, 1), padding='SAME')(out)
        # out += x
        # out = nn.relu(out)

        return out


class UNet(nn.Module):
    in_channels: int
    out_channels: int
    channel_width: int
    depth: int

    @nn.compact
    def __call__(self, inp):

        out = nn.Conv(self.channel_width, kernel_size=(3, 3), padding='SAME')(inp)
        out = nn.relu(out)
        out = nn.Conv(self.channel_width, kernel_size=(3, 3), padding='SAME')(out)
        x = nn.relu(out)

        features = self.channel_width * 2
        skips = []
        for i in range(self.depth):
            x = ResBlock(features, name=f'down_conv_{i}')(x)
            skips.append(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
            features *= 2

        x = ResBlock(features, name='middle_block')(x)
        x = jax.image.resize(x, shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, features), method='bilinear')

        shapes = [a.shape for a in skips]
        print(shapes)
        for i in range(self.depth - 1):
            features //= 2
            x = jnp.concatenate([skips.pop(), x], axis=-1)
            x = ResBlock(features, name=f'up_conv_{i}')(x)
            x = jax.image.resize(x, shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, features), method='bilinear')

        features //= 2
        x = ResBlock(features, name='last_block')(x)

        x = nn.Conv(self.out_channels, kernel_size=(1, 1), name='final_conv')(x)

        return x

shape = (3, 512, 512)
batch_size = 8
epochs = 10000
learning_rate = 1e-7 #1e-3
N = 100
n_points = 5 # 50
num_steps_per_epoch = 1 #N // batch_size
radius = 5#[5, 25]
intensity = 1#[1, 5]
z_planes = [0.0, 0.05, 0.10]

# Example usage
encoder_channels = [64, 128, 512, 1024, 2048]
decoder_channels = [512, 256, 128, 64]#[2048, 1024, 512, 256, 128, 64]
initial_shape = (8, 8, 2048)  # Height x Width x Channels

gen = RandoDotGenerator(N = N,
                        n_points = n_points,
                        radius = radius,
                        intensity = intensity,
                        shape = shape,
                        z_range = [-0.05, 0.05],
                        z_noise = 0.1,
                        background_weight = 0.1,
                        output_arguments = {'x': True,
                                            'y': True,
                                            'z': False,
                                            'radius': False,
                                            'intensity': False,
                                            'image': True,
                                            'pixel_weights': False},
                       z_fixed=z_planes)

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

def get_coord_channels(shape):
    y_vals = jnp.linspace(-1, 1, shape[1])
    x_vals = jnp.linspace(-1, 1, shape[2])
    x_ch, y_ch = jnp.meshgrid(x_vals, y_vals)
    return jnp.stack([x_ch, y_ch], axis=-1)[None, ...]

class CoordConv(nn.Module):
    coord_channels: Array
    num_kernels: int

    @nn.compact
    def __call__(self, inputs):
        channels = jnp.broadcast_to(self.coord_channels, (inputs.shape[0],)+self.coord_channels.shape[1:])
        x = jnp.concatenate([inputs, channels], axis = -1)

        x = nn.Conv(self.num_kernels, kernel_size=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        return x

class PointNetEncoder(nn.Module):
    """Multi-Layer Perceptron (MLP) module used for classification"""
    n_kernels: list[int]

    @nn.compact
    def __call__(self, inputs, train: bool):
        x = nn.Dense(features=self.n_kernels[0])(inputs)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_kernels[1])(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_kernels[2])(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = jnp.max(x, axis=1)
        return x

class DeconvDecoder(nn.Module):
    channels: list
    initial_shape: tuple
    coord_channels: Array

    @staticmethod
    def pixel_shuffle(x, upscale_factor):
        return rearrange(x, 'b h w (c h_scale w_scale) -> b (h h_scale) (w w_scale) c', h_scale=upscale_factor, w_scale=upscale_factor)

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.initial_shape[0]*self.initial_shape[1]*self.channels[0])(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        x = jnp.reshape(x, (-1, self.initial_shape[0], self.initial_shape[1], self.channels[0]))

        for ch in self.channels[1:]:
            x = nn.Conv(ch * 4, kernel_size=(3, 3), padding='SAME')(x)  # Multiply by 4 for 2x2 pixel shuffle (upscale_factor=2)
            x = self.pixel_shuffle(x, upscale_factor=2)
            x = nn.relu(x)

        # for ch in self.channels[1:]:
        #     x = nn.ConvTranspose(ch, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        #     x = nn.relu(x)

        x = CoordConv(self.coord_channels, ch)(x)

        x = nn.Conv(ch, kernel_size=(1, 1), padding='VALID')(x)
        x = nn.relu(x)

        x = nn.Conv(ch, kernel_size=(1, 1), padding='VALID')(x)
        x = nn.relu(x)

        x = nn.Conv(ch * 16, kernel_size=(1, 1), padding='VALID')(x)
        x = nn.relu(x)

        x = nn.Conv(ch * 16, kernel_size=(3, 3), padding='SAME')(x)  # Multiply by 4 for 2x2 pixel shuffle (upscale_factor=2)
        x = self.pixel_shuffle(x, upscale_factor=4)

        x = nn.Conv(ch, kernel_size=(1, 1), padding='VALID')(x)
        x = nn.relu(x)

        x = nn.Conv(2, kernel_size=(1, 1), padding='VALID')(x)
        x = nn.sigmoid(x)

        return x

class NeuralFieldDecoder(nn.Module):
    skip_connection_index: int = 4
    dtype: Any = jnp.float32
    precision: Any = lax.Precision.DEFAULT
    apply_positional_encoding: bool = True
    positional_encoding_dims: int = 6
    num_dense_layers: int = 4 #8
    dense_layer_width: int = 512 #256
    num_output_features: int = 1

    def positional_encoding(self, inputs, encoded, base: float = 2.0):
      batch_size, _ = inputs.shape
      # Applying vmap transform to vectorize the multiplication operation
      inputs_freq = jax.vmap(
          lambda x: inputs * base ** x
      )(jnp.arange(self.positional_encoding_dims))
      periodic_fns = jnp.stack([jnp.sin(inputs_freq), jnp.cos(inputs_freq)])
      periodic_fns = periodic_fns.swapaxes(0, 2).reshape([batch_size, -1])
      periodic_fns = jnp.concatenate([encoded, inputs, periodic_fns], axis=-1)
      return periodic_fns

    @nn.compact
    def __call__(self, input_points, encoded):
        # input_points = input_points * 100
        x = self.positional_encoding(input_points, encoded, base=1.3) if self.apply_positional_encoding else jnp.concatenate([encoded, input_points], axis=-1)
        for i in range(self.num_dense_layers):
            x = nn.Dense(
                self.dense_layer_width,
                dtype=self.dtype,
                precision=self.precision
            )(x)
            freq_and_shift = nn.sigmoid(nn.Dense(
                2,
                dtype=self.dtype,
                precision=self.precision
            )(encoded)) * jnp.pi * 2.0
            freq = freq_and_shift[:, 0, jnp.newaxis]
            shift = freq_and_shift[:, 1, jnp.newaxis]
            x = jnp.sin(freq * x - shift)
            # Skip connection
            x = jnp.concatenate([x, encoded, input_points], axis=-1) if i == 3 else x
        x = nn.Dense(self.num_output_features, dtype=self.dtype, precision=self.precision)(x)
        return x

class MLP(nn.Module):
    hidden_dims: list

    @nn.compact
    def __call__(self, x, train):
        # Hidden layers
        for _ in range(len(self.hidden_dims)):
            x = nn.Dense(self.hidden_dims[_])(x)

            x = nn.relu(x)
        return x

class CedricsGATLayer(nn.Module):
    feats_hidden_dims: list
    adj_hidden_dims: list
    update_hidden_dims: list
    pool: bool = True

    def setup(self):
        # Set up three different MLP instances for different tasks
        self.mlp_feat = MLP(hidden_dims=self.feats_hidden_dims)
        assert self.adj_hidden_dims[-1] == 1, "The last layer of the adjacency MLP should output a single value."
        self.mlp_adj = MLP(hidden_dims=self.adj_hidden_dims)
        self.mlp_update = MLP(hidden_dims=self.update_hidden_dims)

    def __call__(self, coords, train):
        batch_size = coords.shape[0] # this is the batch size
        num_points = coords.shape[1] # this is the batch size

        h = self.mlp_feat(coords, train) # dims should be [batch, num_points, feats]

        # Calculate adjacency matrix using a more efficient batch operation
        expanded_i = jnp.expand_dims(coords, 2)
        expanded_j = jnp.expand_dims(coords, 1)
        adjacency_inputs = jnp.concatenate([expanded_i.repeat(num_points, axis=2), expanded_j.repeat(num_points, axis=1)], axis=-1)
        A = self.mlp_adj(adjacency_inputs.reshape(batch_size, num_points ** 2, -1), train).reshape(batch_size, num_points, num_points)

        # Feature aggregation
        V = jnp.matmul(A, h)
        
        # Update features
        updates = jnp.concatenate([h, coords, V], axis=-1)
        h_next = self.mlp_update(updates, train)

        if self.pool:
            h_next = jnp.max(h_next, axis=1)
        return h_next

def normalize(x): # TODO
    # y = x - jnp.array([0, 0, radius[0], intensity[0]])[None, None, ...]
    y = x - jnp.array([0, 0, 0])[None, None, ...]#, radius[0], intensity[0], z_range[0]])[None, None, ...]
    # y /= jnp.array([shape[2], shape[1], radius[1] - radius[0], intensity[1] - intensity[0]])[None, None, ...]
    y /= jnp.array([shape[2], shape[1], shape[0]])[None, None, ...]#, radius[1] - radius[0], intensity[1] - intensity[0], z_range[1] - z_range[0]])[None, None, ...]
    y -= 0.5
    return y * 2

from flax import linen as nn
from jax import numpy as jnp
import numpy as np
import jax
from jax import lax
from jax import random
from jax import jit
from flax import struct

import jax.numpy as jnp
import flax.linen as nn

class TransformerEncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    dropout_rate: float

    def setup(self):
        self.mha = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim
        )
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.dense1 = nn.Dense(self.embed_dim * 4)
        self.dense2 = nn.Dense(self.embed_dim)
        if self.dropout_rate != 0.0: self.dropout = nn.Dropout(rate=self.dropout_rate)
        else: self.dropout = None

    def __call__(self, x, train: bool = True):
        # Multi-head attention layer
        attn_output = self.mha(x, x, x)
        if self.dropout is not None: attn_output = self.dropout(attn_output, deterministic=not train)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward layer
        ff_output = self.dense1(out1)
        ff_output = nn.relu(ff_output)
        if self.dropout is not None: ff_output = self.dropout(ff_output, deterministic=not train)
        ff_output = self.dense2(ff_output)
        if self.dropout is not None: ff_output = self.dropout(ff_output, deterministic=not train)
        out2 = self.layernorm2(out1 + ff_output)

        return out2

class TransformerEncoder(nn.Module):
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 1
    dropout_rate: float = 0.0

    def setup(self):
        self.input_projection = nn.Dense(self.embed_dim)
        self.encoder_blocks = [TransformerEncoderBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        ) for _ in range(self.num_layers)]

    def __call__(self, x, train: bool = True):
        x = self.input_projection(x)
        for block in self.encoder_blocks:
            x = block(x, train)
        return x.mean(axis=1)

class PointNetImageGen(nn.Module):
    encoder_channels: list
    initial_shape: tuple
    slm_shape: tuple = (shape[1], shape[2]) #(512, 512)
    spacing: float = 9.2e-6
    spectrum: Array = 1.04e-6
    spectral_density: Array = 1.0
    decoder_config: Dict = None

    def setup(self):
      # self.encoder = CedricsGATLayer(feats_hidden_dims=[32, 64, 128], adj_hidden_dims=[32, 64, 128, 1], update_hidden_dims=[256, 512, 1024], pool = True)
      self.encoder = TransformerEncoder()
      self.decoder = NeuralFieldDecoder(self.decoder_config) if self.decoder_config is not None else NeuralFieldDecoder()
      # self.decoder_cleanup = nn.Dense(self.slm_shape[0]*self.slm_shape[1])

    def get_nf_inputs(self, enc):
      slm_points = ScalarField.create(
            self.spacing,
            self.spectrum,
            self.spectral_density,
            shape=self.slm_shape
        ).grid
      slm_points = jnp.stack([slm_points[0].flatten(), slm_points[1].flatten()], axis=-1)
      slm_points = slm_points / slm_points.max()
      repeated_enc = repeat(enc, f'a b -> a {slm_points.shape[0]} b').reshape(-1, enc.shape[-1])
      repeated_slm_points = repeat(slm_points, f"a b -> ({enc.shape[0]} a) b")
      # neural_field_inputs = jnp.concatenate([repeated_enc, repeated_slm_points], axis=-1)
      return (repeated_slm_points, repeated_enc)

    def __call__(self, inputs, train: bool):
        inputs = normalize(inputs)
        encoded = self.encoder(inputs, train) # (b, 512)
        nf_inputs = self.get_nf_inputs(encoded)
        decoded = self.decoder(*nf_inputs)
        # decoded = self.decoder_cleanup(decoded)
        decoded = decoded.reshape(-1, *self.slm_shape)
        return decoded

class DeepCGHEncoder(nn.Module):
  num_features: int = 1024
  
  @nn.compact
  def __call__(self, x):
    embedding = nn.Dense(features=self.num_features)(x)
    embedding = nn.relu(embedding)

    for i in range(6):
       embedding = embedding + nn.MultiHeadAttention(
            num_heads=8,
            qkv_features=self.num_features,
            out_features=self.num_features,
            normalize_qk=True,
        )(embedding)
       embedding = nn.LayerNorm()(embedding)
       embedding = embedding + nn.relu(nn.Dense(features=self.num_features)(embedding))
       embedding = nn.LayerNorm()(embedding)

    return embedding.mean(axis=-2)

class DeepCGHDecoder(nn.Module):
  dtype: Any = jnp.float32
  precision: Any = lax.Precision.DEFAULT
  
  def setup(self):
    self.inr_linear_layers = [nn.Dense(f, dtype=self.dtype, precision=self.precision) for f in [512, 512, 512, 512, 512]]
    self.inr_frequency_shifts = [nn.Dense(2, dtype=self.dtype, precision=self.precision) for f in range(len(self.inr_linear_layers))]
    self.out_layer = nn.Dense(1)

  def positional_encoding(self, inputs, base: float = 2.0):
    inputs_freq = jax.vmap(
        lambda x: inputs * base ** x
    )(jnp.arange(12))
    periodic_fns = jnp.stack([jnp.sin(inputs_freq), jnp.cos(inputs_freq)])
    periodic_fns = periodic_fns.swapaxes(0, 2).reshape([inputs.shape[0], -1])
    periodic_fns = jnp.concatenate([inputs, periodic_fns], axis=-1)
    return periodic_fns

  @partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
  def __call__(self, spot_embedding: Array, slm_points: Array):
    slm_encodings = repeat(slm_points, "c -> b c", b=spot_embedding.shape[0])
    slm_encodings = self.positional_encoding(slm_encodings)
      
    embedding = jnp.concatenate(
          [
              spot_embedding,
              slm_encodings
          ],
          axis=-1
      )
    
    x = embedding
    for i in range(len(self.inr_linear_layers)):
      x = self.inr_linear_layers[i](x)

      frequency_shift = self.inr_frequency_shifts[i](embedding)
      frequency_shift = nn.sigmoid(frequency_shift) * jnp.pi * 2.0

      freq = frequency_shift[:, 0, jnp.newaxis]
      shift = frequency_shift[:, 1, jnp.newaxis]

      x = jnp.sin(freq * x - shift)
    
    x = self.out_layer(x)
    return x

class DeepCGH(nn.Module):
  spacing: float = 9.2e-6
  spectrum: Array = 1.04e-6
  spectral_density: Array = 1.0
  slm_shape: tuple = (shape[1], shape[2])
  bit_depth: int = 8
  f: float = 0.2
  n: float = 1.0
  N_pad: int = 0

  def setup(self):
    self.encoder = DeepCGHEncoder()
    self.decoder = DeepCGHDecoder()

  def get_image_from_phase(self, phase, z):
    field = ScalarField.create(
          self.spacing,
          self.spectrum,
          self.spectral_density,
          rearrange(jnp.exp(1j * phase), "b h w -> b 1 h w 1 1"))
    field = ff_lens(field, self.f, self.n)
    field = transfer_propagate(field, z, self.n, self.N_pad)
    
    recon = rearrange(field.intensity, "b z y x 1 1 -> b z y x")
    return recon

  def __call__(self, xy_coords, z_coords, z_planes, train: bool):
    inputs = jnp.concatenate([xy_coords, z_coords[...,None]], axis=-1)
    inputs = normalize(inputs)
    encoded = self.encoder(inputs)
    
    slm_points = ScalarField.create(
            self.spacing,
            self.spectrum,
            self.spectral_density,
            shape=self.slm_shape
        ).grid
    slm_points = jnp.stack([slm_points[0].flatten(), slm_points[1].flatten()], axis=-1)
    slm_points = slm_points / slm_points.max()

    phase = self.decoder(encoded, slm_points)[:,:,0]
    phase = rearrange(phase, "b (h w) -> b h w", h=self.slm_shape[0], w=self.slm_shape[1])
    phase = quantize(wrap_phase(phase), self.bit_depth, range=(-jnp.pi, jnp.pi))

    recon = self.get_image_from_phase(phase, z_planes)
    
    return (recon, phase), (None, None, None)

@struct.dataclass
class Metrics(metrics.Collection):
  loss_cosine: metrics.Average.from_output('loss_cosine')
  loss_acc: metrics.Average.from_output('loss_acc')

from typing import Any
from collections.abc import Callable

import optax

import jax
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT

class TrainState(train_state.TrainState):
  metrics: Metrics
  batch_stats: Any

  def apply_gradients(self, *, grads, value, **kwargs):
    """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

    Args:
      grads: Gradients that have the same pytree structure as ``.params``.
      **kwargs: Additional dataclass attributes that should be ``.replace()``-ed.

    Returns:
      An updated instance of ``self`` with ``step`` incremented by one, ``params``
      and ``opt_state`` updated by applying ``grads``, and additional attributes
      replaced as specified by ``kwargs``.
    """
    if OVERWRITE_WITH_GRADIENT in grads:
      grads_with_opt = grads['params']
      params_with_opt = self.params['params']
    else:
      grads_with_opt = grads
      params_with_opt = self.params

    updates, new_opt_state = self.tx.update(
      grads_with_opt, self.opt_state, params_with_opt, value=value
    )
    new_params_with_opt = optax.apply_updates(params_with_opt, updates)

    # As implied by the OWG name, the gradients are used directly to update the
    # parameters.
    if OVERWRITE_WITH_GRADIENT in grads:
      new_params = {
        'params': new_params_with_opt,
        OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
      }
    else:
      new_params = new_params_with_opt
    return self.replace(
      step=self.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      **kwargs,
    )

def create_train_state(module, rng, learning_rate):
    """Creates an initial `TrainState`."""
    # initialize parameters by passing a template image
    variables = module.init(rng,
                            jnp.ones((batch_size, n_points, 2)),# if shape[0] == 1 else 5)), # TODO
                            jnp.ones((batch_size, n_points)),
                            jnp.ones_like(np.array(z_planes)),
                            train=False)
    if starting_params_path is None:
        params = variables['params']
    else:
        print('Using starting params')
        params = deserialize('', starting_params_path)

    tx = optax.chain(
        optax.sgd(learning_rate),
        optax.adaptive_grad_clip(clipping=0.07, eps=0.001))
    
    state = TrainState.create(
                            apply_fn=module.apply,
                            params=params,
                            tx=tx,
                            batch_stats=None,
                            metrics=Metrics.empty())
    if starting_opt_state_path is not None:
        print('Using starting opt state')
        state.replace(opt_state=deserialize('', starting_opt_state_path))

    return state

@jax.jit
def train_step(state: TrainState, batch):
    """Train for a single step."""
    def loss_fn(params):
        ((recon, slm_phase), (amp, phase, field)), updates = state.apply_fn({'params': params},# 'batch_stats': state.batch_stats},
                                        xy_coords = batch[0],
                                        z_coords = batch[1],
                                        z_planes = np.array(z_planes),                                   
                                        train=True,
                                        mutable=['batch_stats'])
        loss = loss_cosine(recon, batch[3])
        return loss, (recon, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads, value=loss)
    return state, grads

@jax.jit
def compute_metrics(*, state, batch):
    (recon, slm_phase), (amp, phase, field) = state.apply_fn({'params': state.params},# 'batch_stats': state.batch_stats},
                                    xy_coords = batch[0],
                                    z_coords = batch[1],
                                    z_planes = np.array(z_planes),
                                    train=False)
    loss = {
        'loss_cosine': loss_cosine(recon, batch[3]),
        'loss_acc': loss_acc(recon, batch[3])
    }

    metric_updates = state.metrics.single_from_model_output(predictions = recon, targets=batch[3], loss_cosine=loss['loss_cosine'], loss_acc=loss['loss_acc'])
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state, {'recon': recon, 'slm_phase': slm_phase, 'batch': batch, 'amp': amp, 'phase': phase, 'field': field}

spacing = 9.2e-6
spectrum = 1.04e-6
spectral_density = 1.0

cnn = DeepCGH()

print("Made model!")

init_rng = jax.random.PRNGKey(0)

#%
state = create_train_state(cnn, init_rng, learning_rate)
del init_rng  # Must not be used anymore.

print(jax.tree_map(lambda x: x.shape, state.params))

# Function to calculate the number of parameters
def count_parameters(params):
    return sum(x.size for x in jax.tree_util.tree_flatten(params)[0])

# Function to calculate the model size
def calculate_model_size(params):
    size_in_bytes = sum(x.size * x.itemsize for x in jax.tree_util.tree_flatten(params)[0])
    size_in_mb = size_in_bytes / (1024 ** 2)
    return size_in_mb

num_params = count_parameters(state.params)
model_size = calculate_model_size(state.params)

print(f"Made train state!\nNumber of parameters: {num_params}\nModel size: {model_size:.2f} MB ({(model_size/1024.0):.2f} GB)")

metrics_history = {'train_loss_cosine': [],
                   'train_loss_acc': [],
                   'test_loss_cosine': [],
                   'test_loss_acc': []}

batch = gen.get_batch(batch_size)

for step in range(num_steps_per_epoch*epochs):
  # batch = gen.get_batch(batch_size)
  # Run optimization steps over training batches and compute batch metrics
  if step == 0:
      starting_params = deepcopy(state.params)
  else:
      starting_params = None
      
  state, grads = train_step(state, batch) # get updated train state (which contains the updated parameters)
  state, cache = compute_metrics(state=state, batch=batch) # aggregate batch metrics
  cache['grads'] = grads

  if step == 0:
      grads_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), grads)
      params_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), starting_params)
      norm_ratios = jax.tree_util.tree_map(lambda g, p: g / p, grads_norms, params_norms)
      print("Norm ratios:")
      print(norm_ratios)

  # if (step+1) % 5 == 0: # one training epoch has passed
  #     for metric, value in state.metrics.compute().items():
  #         print(metric, value)

  if (step+1) % num_steps_per_epoch == 0: # one training epoch has passed
    for metric,value in state.metrics.compute().items(): # compute metrics
      metrics_history[f'train_{metric}'].append(value) # record metrics

    if ((step+1) // num_steps_per_epoch) % 50 == 0:
      serialize('', f'current_metrics_history.pkl', metrics_history)
      serialize(save_path, f'current_metrics_history.pkl', metrics_history)
      print('Saved metrics history to current_metrics_history.pkl')

    if ((step+1) // num_steps_per_epoch) % 1000 == 0:
      serialize('', f'curr_met.pkl', metrics_history)
      print('Saved metrics history to curr_met.pkl')

    state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

    if ((step+1) // num_steps_per_epoch) % 1000 == 0:
      serialize(save_path, f'params_epoch_{(step+1) // num_steps_per_epoch}.pkl', state.params)
      print(f'Saved params to params_epoch_{(step+1) // num_steps_per_epoch}.pkl')

      serialize(save_path, f'state_opt_state_epoch_{(step+1) // num_steps_per_epoch}.pkl', state.opt_state)
      print(f'Saved state.opt_state to state_opt_state_epoch_{(step+1) // num_steps_per_epoch}.pkl')

      serialize(save_path, f'cache_epoch_{(step+1) // num_steps_per_epoch}.pkl', cache)
      print(f'Saved cache (recon, slm phase, and batch) to cache_epoch_{(step+1) // num_steps_per_epoch}.pkl')
      
    if ((step+1) // num_steps_per_epoch) % 50 == 0:
      serialize('', f'params_current.pkl', state.params)
      print(f'Saved params to params_current.pkl')

      serialize('', f'state_opt_state_current.pkl', state.opt_state)
      print(f'Saved state.opt_state to state_opt_state_current.pkl')

      serialize('', f'cache_current.pkl', cache)
      print(f'Saved cache (recon, slm phase, and batch) to cache_current.pkl')

    print(f"train epoch: {(step+1) // num_steps_per_epoch}, "
          f"loss_acc: {metrics_history['train_loss_acc'][-1]}, "
          f"loss_cosine: {metrics_history['train_loss_cosine'][-1]}, "
          )
    
datetime = time.strftime("%Y%m%d-%H%M%S")

serialize(save_path, f'metrics_history_{datetime}.pkl', metrics_history)
print(f'metrics_history_{datetime}.pkl')

serialize(save_path, f'params_{datetime}.pkl', state.params)
print(f'params_{datetime}.pkl')

serialize(save_path, f'state_opt_state_{datetime}.pkl', state.opt_state)
print(f'state_opt_state_{datetime}.pkl')
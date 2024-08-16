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

print("Completed imports!!!")
print(jax.devices())

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


def RandoDotTFDataset(N,
                      n_points,
                      radius,
                      intensity,
                      shape,
                      z_range,
                      background_weight,
                      output_arguments,
                      batch_size,
                      epochs):
    '''


    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    n_points : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    intensity : TYPE
        DESCRIPTION.
    shape : TYPE
        DESCRIPTION.
    z_range : TYPE
        DESCRIPTION.
    background_weight : TYPE
        DESCRIPTION.
    output_arguments : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    epochs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    gen = RandoDotGenerator(N = N,
                            n_points = n_points,
                            radius = radius,
                            intensity = intensity,#[1, 10],
                            shape = shape,
                            z_range = z_range,
                            z_noise = 0.1,
                            background_weight = background_weight,
                            output_arguments = output_arguments)

    vector_arguments = {
        'x': output_arguments['x'],
        'y': output_arguments['y'],
        'z': False if shape[0] == 1 else output_arguments['z'], # dummy proofing this
        'radius': output_arguments['radius'],
        'intensity': output_arguments['intensity']}
    image_arguments = {
        'image': output_arguments['image'],
        'pixel_weights': output_arguments['pixel_weights']}

    otype = (tf.float32,) + (tf.float32,) * (sum(image_arguments.values()) + (1 if shape[0] > 1 else 0))
    oshape = (tf.TensorShape((n_points, sum(vector_arguments.values()))),)
    if shape[0] > 1:
        oshape += (tf.TensorShape((shape[0],)),)
    oshape += (tf.TensorShape(shape),) * sum(image_arguments.values())
    # print(otype, oshape)
    ds = tf.data.Dataset.from_generator(gen,
                                        output_types = otype,
                                        output_shapes = oshape)
    return ds.batch(batch_size).repeat(epochs)

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

shape = (1, 1024, 1024)
batch_size = 4 # 32
epochs = 1000
learning_rate = 1e-3
N = 100
n_points = 2 # 50
num_steps_per_epoch = N // batch_size
radius = 5#[5, 25]
intensity = 1#[1, 5]

# Example usage
encoder_channels = [64, 128, 512, 1024, 2048]
decoder_channels = [512, 256, 128, 64]#[2048, 1024, 512, 256, 128, 64]
initial_shape = (8, 8, 2048)  # Height x Width x Channels

train_dset = RandoDotTFDataset(N = N,
                               n_points = n_points,
                               radius = radius,
                               intensity = intensity,
                               shape = shape,
                               z_range = 1,
                               background_weight = 0.1,
                               output_arguments = {'x': True,
                                                   'y': True,
                                                   'z': False,
                                                   'radius': False,
                                                   'intensity': False,
                                                   'image': True,
                                                   'pixel_weights': False},
                               batch_size = batch_size,
                               epochs = epochs)

print("Created train dset!")

def cosine3D(y_true, y_pred):
    intersect = jnp.sum(y_true*y_pred, axis=(1, 2, 3))
    fsum = jnp.sum(y_true**2, axis=(1, 2, 3))
    ssum = jnp.sum(y_pred**2, axis=(1, 2, 3))
    cosine = 1 - jnp.mean((intersect / (fsum * ssum)))
    return cosine

def loss_cosine(predictions, targets):
    return optax.cosine_distance(predictions = predictions.reshape(predictions.shape[0], -1),
                                 targets = targets.reshape(targets.shape[0], -1).astype(jnp.float32),
                                 epsilon=1e-6).mean()

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
        x = self.positional_encoding(input_points, encoded, base=100.0) if self.apply_positional_encoding else jnp.concatenate([encoded, input_points], axis=-1)
        for i in range(self.num_dense_layers):
            x = nn.Dense(
                self.dense_layer_width,
                dtype=self.dtype,
                precision=self.precision
            )(x)
            x = nn.relu(x)
            # Skip connection
            x = jnp.concatenate([x, encoded, input_points], axis=-1) if i == 3 else x
        x = nn.Dense(self.num_output_features, dtype=self.dtype, precision=self.precision)(x)
        # x = jnp.pi * jnp.tanh(x)
        return x

def normalize(x): # TODO
    # y = x - jnp.array([0, 0, radius[0], intensity[0]])[None, None, ...]
    y = x - jnp.array([0, 0])[None, None, ...]#, radius[0], intensity[0], z_range[0]])[None, None, ...]
    # y /= jnp.array([shape[2], shape[1], radius[1] - radius[0], intensity[1] - intensity[0]])[None, None, ...]
    y /= jnp.array([shape[2], shape[1]])[None, None, ...]#, radius[1] - radius[0], intensity[1] - intensity[0], z_range[1] - z_range[0]])[None, None, ...]
    y -= 0.5
    return y * 2

class PointNetImageGen(nn.Module):
    encoder_channels: list
    initial_shape: tuple
    slm_shape: tuple = (128, 128) #(512, 512)
    spacing: float = 9.2e-6
    spectrum: Array = 1.04e-6
    spectral_density: Array = 1.0
    decoder_config: Dict = None

    def setup(self):
      self.encoder = PointNetEncoder(self.encoder_channels)
      self.decoder = NeuralFieldDecoder(self.decoder_config) if self.decoder_config is not None else NeuralFieldDecoder()
      self.decoder_cleanup = nn.Sequential([nn.Conv(32, kernel_size=(5, 5), padding='SAME'),
                                            nn.relu,
                                            nn.Conv(1, kernel_size=(5, 5), padding='SAME')])

    def get_nf_inputs(self, enc):
      slm_points = ScalarField.create(
            self.spacing,
            self.spectrum,
            self.spectral_density,
            shape=self.slm_shape
        ).grid
      slm_points = jnp.stack([slm_points[0].flatten(), slm_points[1].flatten()], axis=-1)
      repeated_enc = repeat(enc, f'a b -> a {slm_points.shape[0]} b').reshape(-1, enc.shape[-1])
      repeated_slm_points = repeat(slm_points, f"a b -> ({enc.shape[0]} a) b")
      # neural_field_inputs = jnp.concatenate([repeated_enc, repeated_slm_points], axis=-1)
      return (repeated_slm_points, repeated_enc)

    def __call__(self, inputs, train: bool):
        inputs = normalize(inputs)
        encoded = self.encoder(inputs, train) # (b, 512)
        nf_inputs = self.get_nf_inputs(encoded)
        decoded = self.decoder(*nf_inputs)
        decoded = decoded.reshape(-1, *self.slm_shape)
        # decoded = decoded.reshape(-1, *self.slm_shape, 1)
        # decoded = self.decoder_cleanup(decoded)[:,:,:,0]
        return decoded

@struct.dataclass
class Metrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
  metrics: Metrics
  batch_stats: Any

def create_train_state(module, rng, learning_rate):
    """Creates an initial `TrainState`."""
    # initialize parameters by passing a template image
    variables = module.init(rng,
                            jnp.ones((batch_size, n_points, 2)),# if shape[0] == 1 else 5)), # TODO
                            train=False)
    params = variables['params']
    batch_stats = variables['batch_stats']
    tx = optax.adam(learning_rate)
    return TrainState.create(
                            apply_fn=module.apply,
                            params=params,
                            tx=tx,
                            batch_stats=batch_stats,
                            metrics=Metrics.empty())


@jax.jit
def train_step(state: TrainState, batch):
    """Train for a single step."""
    def loss_fn(params):
        slm_phase, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                        inputs = batch[0],
                                        train=True,
                                        mutable=['batch_stats'])
        recon = jnp.square(jnp.abs(jnp.fft.fft2(jnp.exp(1j*slm_phase))))[:, None, ...]
        loss = loss_cosine(recon, batch[1])
        # loss = dice(batch[1], recon)# + cosine3D(batch[1], recon)
        return loss, (recon, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state

@jax.jit
def compute_metrics(*, state, batch):
    slm_phase = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats},
                                    inputs = batch[0],
                                    train=False)
    recon = jnp.square(jnp.abs(jnp.fft.fft2(jnp.exp(1j*slm_phase))))[:, None, ...]
    loss = loss_cosine(recon, batch[1])

    metric_updates = state.metrics.single_from_model_output(predictions = recon, targets=batch[1], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

spacing = 9.2e-6
spectrum = 1.04e-6
spectral_density = 1.0

cnn = PointNetImageGen(encoder_channels=encoder_channels, initial_shape=initial_shape, slm_shape=(shape[1], shape[2]), spacing=spacing, spectrum=spectrum, spectral_density=spectral_density, decoder_config=None)

print("Made model!")

init_rng = jax.random.PRNGKey(0)

#%
state = create_train_state(cnn, init_rng, learning_rate)
del init_rng  # Must not be used anymore.

print("Created train state!")

metrics_history = {'train_loss': [],
                   'test_loss': []}

batch = train_dset.as_numpy_iterator().__next__()
print("Got first batch!")
for step in range(num_steps_per_epoch*epochs):
  print(f"Beginning step {step+1}")
  # Run optimization steps over training batches and compute batch metrics
  state = train_step(state, batch) # get updated train state (which contains the updated parameters)
  state = compute_metrics(state=state, batch=batch) # aggregate batch metrics
  print(f"Completed step {step+1}")
  if (step+1) % 5 == 0: # one training epoch has passed
      for metric, value in state.metrics.compute().items():
          print(metric, value)

  if (step+1) % num_steps_per_epoch == 0: # one training epoch has passed
    for metric,value in state.metrics.compute().items(): # compute metrics
      metrics_history[f'train_{metric}'].append(value) # record metrics
    state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

    print(f"train epoch: {(step+1) // num_steps_per_epoch}, "
          f"loss: {metrics_history['train_loss'][-1]}, "
          )
    
datetime = time.strftime("%Y%m%d-%H%M%S")
pickle.dump(metrics_history, open(f'metrics_history_{datetime}.pkl', 'wb'))
pickle.dump(state.params, open(f'params_{datetime}.pkl', 'wb'))
pickle.dump(state.batch_stats, open(f'batch_stats_{datetime}.pkl', 'wb'))

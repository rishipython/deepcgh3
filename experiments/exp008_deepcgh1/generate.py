import jax
from jax.experimental.host_callback import id_print
import jax.numpy as jnp
import numpy as np
import optax
from optax import adam
from flax import linen as nn
from flax.core import frozen_dict
from chex import Array, ArrayTree
import tensorflow as tf
import warnings
from skimage.draw import disk, line_aa
import zarr

import pickle
import os, sys, datetime
from functools import partial
from typing import *
from einops import rearrange, repeat

# from fouriernet.jax import *
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
# from holoscope.optics import *
# from holoscope.optimization import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# save_path = f"/nrs/turaga/debd/lfm/holoscope-experiments/{os.getcwd().split('/')[-2]}/{os.getcwd().split('/')[-1]}"
save_path = '/nrs/turaga/athavaler/experiments/exp008_deepcgh1'

os.makedirs(save_path, exist_ok=True)

class DeepCGH_Datasets(object):
    '''
    Class for the Dataset object used in DeepCGH algorithm.
    Inputs:
        path   string, determines the lcoation that the datasets are going to be stored in
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self, params):
        try:
            assert params['object_type'] in ['Disk', 'Line', 'Dot'], 'Object type not supported'
            
            self.path = params['path']
            self.shape = params['shape']
            self.N = params['N']
            self.ratio = params['train_ratio']
            self.object_size = params['object_size']
            self.intensity = params['intensity']
            self.object_count = params['object_count']
            self.name = params['name']
            self.object_type = params['object_type']
            self.centralized = params['centralized']
            self.normalize = params['normalize']
            self.compression = params['compression']
        except:
            assert False, 'Not all parameters are provided!'
            
        self.__check_avalability()
        
    
    def __check_avalability(self):
        print('Current working directory is:')
        print(os.getcwd(),'\n')
        
        self.filename = self.object_type + '_SHP{}_N{}_SZ{}_INT{}_Crowd{}_CNT{}_Split.tfrecords'.format(self.shape, 
                                           self.N, 
                                           self.object_size,
                                           self.intensity, 
                                           self.object_count,
                                           self.centralized)
        
        self.absolute_file_path = os.path.join(self.path, self.filename)
        if not (os.path.exists(self.absolute_file_path.replace('Split', '')) or os.path.exists(self.absolute_file_path.replace('Split', 'Train')) or os.path.exists(self.absolute_file_path.replace('Split', 'Test'))):
            warnings.warn('File does not exist. New dataset will be generated once getDataset is called.')
            print(self.absolute_file_path)
        else:
            print('Data already exists.')
           
            
    def __get_line(self, shape, start, end):
        img = np.zeros(shape, dtype=np.float32)
        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
        img[rr, cc] = val * 1
        return img
    
    
    def get_circle(self, shape, radius, location):
        """Creates a single circle.
    
        Parameters
        ----------
        shape : tuple of ints
            Shape of the output image
        radius : int
            Radius of the circle.
        location : tuple of ints
            location (x,y) in the image
    
        Returns
        -------
        img
            a binary 2D image with a circle inside
        rr2, cc2
            the indices for a circle twice the size of the circle. This is will determine where we should not create circles
        """
        img = np.zeros(shape, dtype=np.float32)
        rr, cc = disk((location[0], location[1]), radius, shape=img.shape)
        img[rr, cc] = 1
        # get the indices that are forbidden and return it
        rr2, cc2 = disk((location[0], location[1]), 2*radius, shape=img.shape)
        return img, rr2, cc2


    def __get_allowables(self, allow_x, allow_y, forbid_x, forbid_y):
        '''
        Remove the coords in forbid_x and forbid_y from the sets of points in
        allow_x and allow_y.
        '''
        for i in forbid_x:
            try:
                allow_x.remove(i)
            except:
                continue
        for i in forbid_y:
            try:
                allow_y.remove(i)
            except:
                continue
        return allow_x, allow_y
    
    
    def __get_randomCenter(self, allow_x, allow_y):
        list_x = list(allow_x)
        list_y = list(allow_y)
        ind_x = np.random.randint(0,len(list_x))
        ind_y = np.random.randint(0,len(list_y))
        return list_x[ind_x], list_y[ind_y]
    
    
    def __get_randomStartEnd(self, shape):
        start = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        end = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        return start, end


    #% there shouldn't be any overlap between the two circles 
    def __get_RandDots(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random dots
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        xs = list(np.random.randint(0, shape[0], (n,)))
        ys = list(np.random.randint(0, shape[1], (n,)))
        
        for x, y in zip(xs, ys):
            image[x, y] = 1
            
        return image

    #% there shouldn't be any overlap between the two circles 
    def __get_RandLines(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random lines
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        for i in range(n):
            # generate centers
            start, end = self.__get_randomStartEnd(shape)
            
            # get circle
            img = self.__get_line(shape, start, end)
            image += img
        image -= image.min()
        image /= image.max()
        return image
    
    #% there shouldn't be any overlap between the two circles 
    def __get_RandBlobs(self, shape, maxnum = [10,12], radius = 5, intensity = 1):
        '''
        returns a single sample (2D image) with random blobs
        '''
        # random number of blobs to be generated
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        try: # in case the radius of the blobs is variable, get the largest diameter
            r = radius[-1]
        except:
            r = radius
        
        # define sets for storing the values
        allow_x = set(range(shape[0]))
        allow_y = set(range(shape[1]))
        if not self.centralized:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])))
        else:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])) + list(range(shape[0]//6, (5)*shape[0]//6)))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])) + list(range(shape[1]//6, (5)*shape[1]//6)))
        
        allow_x, allow_y = self.__get_allowables(allow_x, allow_y, forbid_x, forbid_y)
        count = 0
        # else
        for i in range(n):
            # generate centers
            x, y = self.__get_randomCenter(allow_x, allow_y)
            
            if isinstance(radius, list):
                r = int(np.random.randint(radius[0], radius[1]))
            else:
                r = radius
            
            if isinstance(intensity, list):
                int_4_this = int(np.random.randint(np.round(intensity[0]*100), np.round(intensity[1]*100)))
                int_4_this /= 100.
            else:
                int_4_this = intensity
            
            # get circle
            img, xs, ys = self.get_circle(shape, r, (x,y))
            allow_x, allow_y = self.__get_allowables(allow_x, allow_y, set(xs), set(ys))
            image += img * int_4_this
            count += 1
            if len(allow_x) == 0 or len(allow_y) == 0:
                break
        return image
    
    
    def coord2image(self, coords):
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            canvas = np.zeros(self.shape[:-1], dtype=np.float32)
        
            for i in range(coords.shape[-1]):
                img, _, __ = self.get_circle(self.shape[:-1], self.object_size, [coords[0, i], coords[1, i]])
                canvas += img.astype(np.float32)
            
            sample[:, :, plane] = (canvas>0)*1.
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
            
        sample -= sample.min()
        sample /= sample.max()
        
        return np.expand_dims(sample, axis = 0)
                
    
    def get_randSample(self):
        
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            if self.object_type == 'Disk':
                img = self.__get_RandBlobs(shape = (self.shape[0], self.shape[1]),
                                           maxnum = self.object_count,
                                           radius = self.object_size,
                                           intensity = self.intensity)
            elif self.object_type == 'Line':
                img = self.__get_RandLines((self.shape[0], self.shape[1]),
                                           maxnum = self.object_count)
            elif self.object_type == 'Dot':
                img = self.__get_RandDots(shape = (self.shape[0], self.shape[1]),
                                          maxnum = self.object_count)
                

            sample[:, :, plane] = img
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
        
        sample -= sample.min()
        sample /= sample.max()
        
        return sample
    
    
    def __bytes_feature(self, value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
    
    
    def __int64_feature(self, value):
      return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
    
    def __generate(self):
        '''
        Creates a dataset of randomly located blobs and stores the data in an TFRecords file. Each sample (3D image) contains
        a randomly determined number of blobs that are randomly located in individual planes.
        Inputs:
            filename : str
                path to the dataset file
            N: int
                determines the number of samples in the dataset
            fraction : float
                determines the fraction of N that is used as "train". The rest will be the "test" data
            shape: (int, int)
                tuple of integers, shape of the 2D planes
            maxnum: int
                determines the max number of blobs
            radius: int
                determines the radius of the blobs
            intensity : float or [float, float]
                intensity of the blobs. If a scalar, it's a binary blob. If a list, first element is the min intensity and
                second one os the max intensity.
            normalize : bool
                flag that determines whether the 3D data is normalized for fixed energy from plane to plane
    
        Outputs:
            aa:
    
            out_dataset:
                numpy.ndarray. Numpy array with shape (samples, x, y)
        '''
        
#        assert self.shape[-1] > 1, 'Wrong dimensions {}. Number of planes cannot be {}'.format(self.shape, self.shape[-1])
        
        train_size = np.floor(self.ratio * self.N)
        # TODO multiple tfrecords files to store data on. E.g. every 1000 samples in one file
        options = tf.io.TFRecordOptions(compression_type = self.compression)
        zarr_train = zarr.open(f"{self.path}/_SHP{self.shape}_N{self.N}_SZ{self.object_size}_INT{self.intensity}_Crowd{self.object_count}_CNT{self.centralized}/train.zarr", shape=(int(train_size), *self.shape), chunks=(1, *self.shape), mode="w")
        zarr_test = zarr.open(f"{self.path}/_SHP{self.shape}_N{self.N}_SZ{self.object_size}_INT{self.intensity}_Crowd{self.object_count}_CNT{self.centralized}/test.zarr", shape=(int(self.N - train_size), *self.shape), chunks=(1, *self.shape), mode="w")
        train_counter = 0
        test_counter = 0
#        options = None
        with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split', 'Train'), options = options) as writer_train:
            with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split', 'Test'), options = options) as writer_test:
                for i in range(self.N):
                    print(f"generating {i + 1} / {self.N}...")
                    sample = self.get_randSample()
                    
                    image_raw = sample.tostring()
                    
                    feature = {'sample': self.__bytes_feature(image_raw)}
                    
                    # 2. Create a tf.train.Features
                    features = tf.train.Features(feature = feature)
                    # 3. Createan example protocol
                    example = tf.train.Example(features = features)
                    # 4. Serialize the Example to string
                    example_to_string = example.SerializeToString()
                    # 5. Write to TFRecord
                    if i < train_size:
                        writer_train.write(example_to_string)
                        zarr_train[train_counter] = sample
                        train_counter += 1
                    else:
                        writer_test.write(example_to_string)
                        zarr_test[test_counter] = sample
                        test_counter += 1
            
    
    def getDataset(self):
        if not (os.path.exists(self.absolute_file_path.replace('Split', '')) or os.path.exists(self.absolute_file_path.replace('Split', 'Train')) or os.path.exists(self.absolute_file_path.replace('Split', 'Test'))):
            print('Generating data...')
            folder = self.path
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            self.__generate()
        self.dataset_paths = [self.absolute_file_path.replace('Split', 'Train'), self.absolute_file_path.replace('Split', 'Test')]


# dataset = DeepCGH_Datasets(
#     {
#         'path' : save_path,
#         'shape' : (512, 512, 3),
#         'object_type' : 'Disk',
#         'object_size' : 10,
#         'object_count' : [27, 48],
#         'intensity' : [0.1, 1],
#         'normalize' : True,
#         'centralized' : False,
#         'N' : 2560,
#         'train_ratio' : 2048/2560,
#         'compression' : 'GZIP',
#         'name' : 'target',
#     }
# )
# dataset.getDataset()


dataset = DeepCGH_Datasets(
    {
        'path' : save_path,
        'shape' : (128, 128, 1),
        'object_type' : 'Disk',
        'object_size' : 5,
        'object_count' : [5, 6],
        'intensity' : 1,
        'normalize' : True,
        'centralized' : False,
        'N' : 2560,
        'train_ratio' : 2048/2560,
        'compression' : 'GZIP',
        'name' : 'target',
    }
)
dataset.getDataset()

# dataset = DeepCGH_Datasets(
#     {
#         'path' : save_path,
#         'shape' : (2048, 2048, 3),
#         'object_type' : 'Disk',
#         'object_size' : 10,
#         'object_count' : [27, 48],
#         'intensity' : [0.1, 1],
#         'normalize' : True,
#         'centralized' : False,
#         'N' : 2560,
#         'train_ratio' : 2048/2560,
#         'compression' : 'GZIP',
#         'name' : 'target',
#     }
# )
# dataset.getDataset()
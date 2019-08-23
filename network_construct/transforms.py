
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import random
import torch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MRISelect(object):
    """Convert ndarrays in img to Tensors."""

    def __call__(self, sample):
        image, label = sample
        img=image.numpy()
        f_length = img.shape[0]
        num_segments = 116
        unit_len = f_length / (num_segments + 1)
        offsets = np.multiply(np.arange(num_segments) + np.random.random(num_segments), unit_len) 
        offsets = offsets.round().astype(np.int)
        return [img[offsets,:,:,:], label]

class ChannelSelect(object):
    """select 64 input channels for ROI time series data."""

    def __call__(self, sample):
        img, roi, label=sample
        img=img.numpy()
        r_length = roi.shape[0]
        f_length = img.shape[0]
        num_segments = 116
        unit_len = r_length / (num_segments + 1)
        offsets = np.multiply(np.arange(num_segments) + np.random.random(num_segments), unit_len) 
        offsets = offsets.round().astype(np.int)
        
        return [img[offsets,:,:,:], roi[offsets], label]
        
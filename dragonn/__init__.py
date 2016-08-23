#!/usr/bin/env python

import os
import numpy as np

# Assumes we are using a g2.8xlarge with 4 GPUs
gpu_idx = np.random.randint(0, 4)

os.putenv('CUDA_VISIBLE_DEVICES', str(gpu_idx))


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import time
import copy
from core50.dataset import CORE50
import torch
import numpy as np
from utils.train_test import train_net, test_multitask, preprocess_imgs
import torchvision.models as models
from utils.common import create_code_snapshot
import tensorflow as tf
import numpy as np
from DEN import DEN


np.random.seed(1004)
flags = tf.app.flags
flags.DEFINE_integer("max_iter", 4300, "Epoch to train")
flags.DEFINE_float("lr", 0.001, "Learing rate(init) for train")
flags.DEFINE_integer("batch_size", 16, "The size of batch for 1 iteration")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory path to save the checkpoints")
flags.DEFINE_list("dims", [784, 312, 128, 50], "Dimensions about layers including output")
flags.DEFINE_integer("n_classes", 50, 'The number of classes at each task')
flags.DEFINE_float("l1_lambda", 0.00001, "Sparsity for L1")
flags.DEFINE_float("l2_lambda", 0.0001, "L2 lambda")
flags.DEFINE_float("gl_lambda", 0.001, "Group Lasso lambda")
flags.DEFINE_float("regular_lambda", 0.5, "regularization lambda")
flags.DEFINE_integer("ex_k", 10, "The number of units increased in the expansion processing")
flags.DEFINE_float('loss_thr', 0.01, "Threshold of dynamic expansion")
flags.DEFINE_float('spl_thr', 0.05, "Threshold of split and duplication")
FLAGS = flags.FLAGS
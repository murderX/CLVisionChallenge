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
from DEN.DEN import DEN

flags = tf.app.flags
flags.DEFINE_integer("max_iter", 4300, "Epoch to train")
flags.DEFINE_float("lr", 0.001, "Learing rate(init) for train")
flags.DEFINE_integer("batch_size", 4, "The size of batch for 1 iteration")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory path to save the checkpoints")
flags.DEFINE_list("dims", [512, 256, 128, 64, 50], "Dimensions about layers including output")
flags.DEFINE_integer("n_classes", 50, 'The number of classes at each task')
flags.DEFINE_float("l1_lambda", 0.00001, "Sparsity for L1")
flags.DEFINE_float("l2_lambda", 0.0001, "L2 lambda")
flags.DEFINE_float("gl_lambda", 0.001, "Group Lasso lambda")
flags.DEFINE_float("regular_lambda", 0.5, "regularization lambda")
flags.DEFINE_integer("ex_k", 10, "The number of units increased in the expansion processing")
flags.DEFINE_float('loss_thr', 0.01, "Threshold of dynamic expansion")
flags.DEFINE_float('spl_thr', 0.05, "Threshold of split and duplication")
FLAGS = flags.FLAGS


def main(args):
    # print args recap
    # print(args, end="\n\n")

    # do not remove this line
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(root='core50/data/', scenario="multi-task-nc",
                     preload=True)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()

    # model
    # if args.classifier == 'ResNet18':
    #     classifier = models.resnet18(pretrained=True)
    #     classifier.fc = torch.nn.Linear(512, args.n_classes)
    model = DEN(FLAGS)
    clf = models.resnet18(pretrained=True)
    clf.fc = torch.nn.Linear(512, 512)
    params = dict()
    avg_perf = []
    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []
    ext_mem = None

    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch
        print('Begin Task ' + str(i))
        train_x = train_x.reshape((len(train_x), 128 * 128 * 3))[:2000]
        train_y = train_y.reshape((1, len(train_y))).astype(np.long)[:2000]
        # One hot encoding
        train_y = np.eye(FLAGS.n_classes)[train_y]
        val_x = full_valdidset[i][0][0]
        val_x = val_x.reshape((len(val_x), 128 * 128 * 3))
        val_y = full_valdidset[i][0][1].astype(np.long)
        val_y = val_y.reshape((1, len(val_y)))
        val_y = np.eye(FLAGS.n_classes)[val_y]
        data = [train_x, train_y, val_x, val_y, val_x, val_y]

        model.sess = tf.Session()
        print("\n\n\tTASK %d TRAINING\n" % (i + 1))

        model.T = model.T + 1
        model.task_indices.append(i + 1)
        model.load_params(params, time=1)
        perf, sparsity, expansion = model.add_task(i + 1, data)

        params = model.get_params()
        model.destroy_graph()
        model.sess.close()

        # model.sess = tf.Session()
        # print('\n OVERALL EVALUATION')
        # model.load_params(params)
        # temp_perfs = []
        # # run test
        # for j in range(t + 1):
        #     temp_perf = model.predict_perform(j + 1, testXs[j], mnist.test.labels)
        #     temp_perfs.append(temp_perf)
        # avg_perf.append(sum(temp_perfs) / float(t + 1))
        # print("   [*] avg_perf: %.4f" % avg_perf[t])
        # model.destroy_graph()
        # model.sess.close()


if __name__ == '__main__':
    main(dict())

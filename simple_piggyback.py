from core50.dataset import CORE50
import torch
import numpy as np
from utils.train_piggyback import train_net, test_multitask
from utils.train_test import preprocess_imgs
import torchvision.models as models
from utils.common import create_code_snapshot
import utils as utils
from piggyback.networks import ModifiedResNet
from torch import optim
import time
import argparse
import os
import copy
from piggyback import utils as utils
import warnings


BATCH_SIZE = 32


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self, args):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []
        self.args = args

    def add(self, optimizer, lr, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(lr)
        self.decay_every.append(decay_every)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = utils.step_lr(
                epoch_idx, init_lr, decay_every,
                0.1, optimizer)


def main(args):
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(root='core50/data/', scenario=args.scenario,
                     preload=args.preload_data)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()
    model = ModifiedResNet()

    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []
    ext_mem = None

    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch

        # adding eventual replay patterns to the current batch
        idxs_cur = np.random.choice(
            train_x.shape[0], args.replay_examples, replace=False
        )

        if i == 0:
            ext_mem = [train_x[idxs_cur], train_y[idxs_cur]]
        else:
            ext_mem = [
                np.concatenate((train_x[idxs_cur], ext_mem[0])),
                np.concatenate((train_y[idxs_cur], ext_mem[1]))]

        train_x = np.concatenate((train_x, ext_mem[0]))
        train_y = np.concatenate((train_y, ext_mem[1]))

        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)

        model.add_dataset([train_x, train_y], 50)
        model.set_dt(i)
        optimizer_masks = optim.Adam(
            model.shared.parameters(), lr=1e-4)
        optimizer_classifier = optim.Adam(
            model.classifier.parameters(), lr=1e-4)

        optimizers = Optimizers(args)
        optimizers.add(optimizer_masks, 1e-4,
                       5)
        optimizers.add(optimizer_classifier, 1e-4,
                       5)

        _, _, stats = train_net(
            args, optimizers, model, criterion, BATCH_SIZE, train_x, train_y, t,
            args.epochs, preproc=preprocess_imgs
        )

        if args.scenario == "multi-task-nc":
            heads.append(copy.deepcopy(model.classifier))

        # collect statistics
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']

        # test on the validation set
        stats, _ = test_multitask(
            model, full_valdidset, args.batch_size,
            preproc=preprocess_imgs, multi_heads=heads, verbose=False
        )

        valid_acc += stats['acc']
        print("------------------------------------------")
        print("Avg. acc: {}".format(stats['acc']))
        print("------------------------------------------")

    # Generate submission.zip
    # directory with the code snapshot to generate the results
    sub_dir = 'submissions/' + args.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    elapsed = (time.time() - start) / 60
    print("Training Time: {}m".format(elapsed))
    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    # test_preds.txt: with a list of labels separated by "\n"
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()
    stats, preds = test_multitask(
        model, full_testset, args.batch_size, preproc=preprocess_imgs,
        multi_heads=heads, verbose=False
    )

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    print("Experiment completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="multi-task-nc",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet50',
                        choices=['ResNet18', 'ResNet50'])

    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')

    # Continual Learning
    parser.add_argument('--replay_examples', type=int, default=0,
                        help='data examples to keep in memory for each batch '
                             'for replay.')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="multi-task-nc",
                        help='directory of the submission file for this exp.')

    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    main(args)

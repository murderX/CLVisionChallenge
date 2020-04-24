import numpy as np
import torch
from torch.autograd import Variable
from .common import pad_data, shuffle_in_unison, check_ext_mem, check_ram_usage


def train_net(args, optimizer, model, criterion, mb_size, x, y, t,
              train_ep, preproc=None, use_cuda=True, mask=None):
    cur_ep = 0
    cur_train_t = t
    stats = {"ram": [], "disk": []}

    if preproc:
        x = preproc(x)

    (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
    )

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = model.cuda()
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    for ep in range(train_ep):
        # Do Epoch
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())
        model.train()
        optimizer.update_lr(ep)
        # if args.train_bn:
        #     model.train()
        # else:
        model.train_nobn()
        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it in range(it_x_ep):
            # Do batch
            start = it * mb_size
            end = (it + 1) * mb_size
            model.zero_grad()
            x_mb = train_x[start:end].cuda()
            y_mb = train_y[start:end].cuda()
            x_mb = Variable(x_mb)
            y_mb = Variable(y_mb)
            output = model(x_mb)
            loss = criterion(output, y_mb)
            loss.backward()

            # Scale gradients by average weight magnitude.
            # if args.mask_scale_gradients != 'none':
            #     for module in model.shared.modules():
            #         if 'ElementWise' in str(type(module)):
            #             abs_weights = module.weight.data.abs()
            #             if args.mask_scale_gradients == 'average':
            #                 module.mask_real.grad.data.div_(abs_weights.mean())
            #             elif args.mask_scale_gradients == 'individual':
            #                 module.mask_real.grad.data.div_(abs_weights)

            # Set batchnorm grads to 0, if required.
            # if not args.train_bn:
            #     for module in model.shared.modules():
            #         if 'BatchNorm' in str(type(module)):
            #             if module.weight.grad is not None:
            #                 module.weight.grad.data.fill_(0)
            #             if module.bias.grad is not None:
            #                 module.bias.grad.data.fill_(0)

            # Update params.
            optimizer.step()
            ave_loss += loss.item()
            _, pred_label = torch.max(output, 1)
            correct_cnt += (pred_label == y_mb).sum()

            acc = correct_cnt.item() / ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))
            # print(
            #     '==>>> it: {}, avg. loss: {:.6f}, '
            #     'running train acc: {:.3f}'
            #         .format(it, ave_loss, acc)
            # )
            if it % 100 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )
        print('Num 0ed out parameters:')
        for idx, module in enumerate(model.shared.modules()):
            if 'ElementWise' in str(type(module)):
                num_zero = module.mask_real.data.lt(5e-3).sum()
                total = module.mask_real.data.numel()
                print(idx, num_zero, total)
        # if args.threshold_fn == 'binarizer':
        #     print('Num 0ed out parameters:')
        #     for idx, module in enumerate(model.shared.modules()):
        #         if 'ElementWise' in str(type(module)):
        #             num_zero = module.mask_real.data.lt(5e-3).sum()
        #             total = module.mask_real.data.numel()
        #             print(idx, num_zero, total)
        # elif args.threshold_fn == 'ternarizer':
        #     print('Num -1, 0ed out parameters:')
        #     for idx, module in enumerate(model.shared.modules()):
        #         if 'ElementWise' in str(type(module)):
        #             num_neg = module.mask_real.data.lt(0).sum()
        #             num_zero = module.mask_real.data.lt(5e-3).sum() - num_neg
        #             total = module.mask_real.data.numel()
        #             print(idx, num_neg, num_zero, total)
        cur_ep += 1
    print('-' * 20)
    return ave_loss, acc, stats



def test_multitask(
        model, test_set, mb_size, preproc=None, use_cuda=True, multi_heads=[], verbose=True):
    """
    Test a model considering that the test set is composed of multiple tests
    one for each task.

        Args:
            model (nn.Module): the pytorch model to test.
            test_set (list): list of (x,y,t) test tuples.
            mb_size (int): mini-batch size.
            preproc (func): image preprocess function.
            use_cuda (bool): if we want to use gpu or cpu.
            multi_heads (list): ordered list of "heads" to be used for each
                                task.
        Returns:
            stats (float): collected stasts of the test including average and
                           per class accuracies.
    """

    model.eval()

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:

        if preproc:
            x = preproc(x)

        if multi_heads != [] and len(multi_heads) > t:
            # we can use the stored head
            if verbose:
                print("Using head: ", t)
            with torch.no_grad():
                model.classifier.weight.copy_(multi_heads[t].weight)
                model.classifier.bias.copy_(multi_heads[t].bias)

        model = model.cuda()
        acc = None

        test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            iters = test_y.size(0) // mb_size + 1
            for it in range(iters):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = test_x[start:end].cuda()
                y_mb = test_y[start:end].cuda()
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()
                preds += list(pred_label.data.cpu().numpy())

                # print(pred_label)
                # print(y_mb)
            acc = correct_cnt.item() / test_y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    if verbose:
        print("------------------------------------------")
        print("Avg. acc:", stats['acc'])
        print("------------------------------------------")

    # reset the head for the next batch
    if multi_heads:
        if verbose:
            print("classifier reset...")
        with torch.no_grad():
            model.classifier.weight.fill_(0)
            model.classifier.bias.fill_(0)

    return stats, preds

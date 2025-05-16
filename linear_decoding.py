import argparse
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from osiris_model import SimSiamSupCon

import wandb
import apex
import torch.nn.functional as F
import math

def set_parameter_requires_grad(model):
    '''Helper function for setting body to non-trainable'''
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        else:
            print(f"{name} is trainable")

def load_split_train_test(datadir, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    test_transforms = transforms.Compose([
                transforms.Resize(256 * args.image_size // 224),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                normalize,
            ])

    # train_transforms = test_transforms = transforms.Compose([transforms.Resize(size=(args.image_size, args.image_size)), transforms.ToTensor(), normalize])

    train_data = datasets.ImageFolder(os.path.join(datadir, 'train'), transform=train_transforms)
    test_data = datasets.ImageFolder(os.path.join(datadir, 'val'), transform=test_transforms)

    num_train = len(train_data)
    num_test = len(test_data)

    train_idx = list(range(num_train))
    test_idx = list(range(num_test))

    print('Training data size is', len(train_idx))
    print('Test data size is', len(test_idx))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return trainloader, testloader

def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate based on schedule"""
    lr = 3.0
    epochs = 10
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(args):
    model = SimSiamSupCon(total_batch_size=512, replay_batch_size=448, group_norm=True, depth=50, curr_loss_coef=1.0, tc_loss_coef=1.0, tc_start_epoch=0, tc_curr_coef=0, fc_eval_numclasses=args.num_classes).cuda()

    if args.fc_bn:
        class L2Norm(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return F.normalize(x, 2)

        model.fc = nn.Sequential(L2Norm(), model.fc)

    # model = SimCLRSupCon(total_batch_size=512, replay_batch_size=448, group_norm=True, depth=50, curr_loss_coef=1.0, tc_loss_coef=1.0, tc_start_epoch=0, tc_curr_coef=0.0, fc_eval_numclasses=args.num_classes).cuda()

    try:
        checkpoint = torch.load(args.load_dir)['model_state_dict']
        all_keys = list(checkpoint.keys())
        for k in all_keys:
            if k.startswith('predictor'):
                del checkpoint[k]
        # import pdb; pdb.set_trace()
        model.load_state_dict(checkpoint, strict=False)

    except KeyError:
        # DEBUGGING
        checkpoint = torch.load(args.load_dir, weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
    
    set_parameter_requires_grad(model)  # freeze the trunk

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(model.fc.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    optimizer = torch.optim.SGD(model.fc.parameters(), lr=3.0, momentum=0.9, weight_decay=0.0)
    optimizer = apex.parallel.LARC.LARC(optimizer, trust_coefficient=0.001, clip=False)

    train_loader, test_loader = load_split_train_test(args.data, args)
    acc1_list = []
    val_acc1_list = []

    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
        acc1_list.append(acc1)
        wandb.log({"train_acc1": acc1}, step=epoch)

        # validate at end of epoch
        val_acc1 = validate(test_loader, model, args)
        val_acc1_list.append(val_acc1)
        wandb.log({"val_acc1": val_acc1}, step=epoch)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model.eval_forward(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model.eval_forward(images)

            preds = np.argmax(output.cpu().numpy(), axis=1)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear decoding with headcam data')
    parser.add_argument('--data', help='path to dataset')
    parser.add_argument('--load_dir', type=int, help='model checkpoint path')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--image_size', default=112, type=int, help='size of image')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1024, type=int, help='mini-batch size (default: 1024)')
    parser.add_argument('--lr', '--learning_rate', default=0.0005, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=0.0, type=float, help='weight decay (default: 0)', dest='weight_decay')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency (default: 100)')
    parser.add_argument('--num_classes', default=26, type=int, help='number of classes in downstream classification task')
    parser.add_argument('--fc_bn', action='store_true', default=False, help='whether to use L2 norm before FC')

    args = parser.parse_args()

    model_ckpt_parts = args.load_dir.split('/')
    if "Saycam-SSL" in args.load_dir:
        args.name = 'imagenet_' + model_ckpt_parts[-1][:-4]
    else:
        args.name = 'imagenet_' + model_ckpt_parts[-2]

    wandb.init(project="saycam-ssl", name=args.name, config=args)

    main(args)

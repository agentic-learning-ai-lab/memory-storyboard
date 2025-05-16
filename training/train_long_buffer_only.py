import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
from osiris_model import SimSiamSupCon, SimCLRTC, SimCLRSupCon
import wandb
import argparse
from datetime import datetime
from dataset_utils import *
from eval_model import eval_imagenet, eval_labeledS
from replay_buffer import ReservoirBatchSampler
from dataset_newsampler import StreamingDataset

wandb.require("core")

class Trainer():
    def __init__(self, args):

        self.args = args
        self.global_step = 0

        num_workers = int(len(os.sched_getaffinity(0))) # num CPUs on the machine
        self.num_workers = num_workers

        self.longmem_dataset = StreamingDataset(dataset=args.dataset, scale_min=args.scale_min, scale_max=args.scale_max, ensure_overlap=args.ensure_overlap, subsample=args.dataset_subsample, class_length=args.class_length, crop_size=self.args.crop_size)

        if args.method == 'tc':
            num_total_classes = len(self.longmem_dataset) // (args.class_length // args.dataset_subsample) + 1
            self.model = SimCLRTC(total_batch_size=args.replay_batch_size, replay_batch_size=args.replay_batch_size, group_norm=args.group_norm, depth=args.depth, curr_loss_coef=0, tc_loss_coef=args.tc_loss_coef, num_total_classes=num_total_classes, tc_start_epoch=0)
        elif args.method == 'simclr':
            self.model = SimCLRSupCon(total_batch_size=args.replay_batch_size, replay_batch_size=args.replay_batch_size, group_norm=args.group_norm, depth=args.depth, curr_loss_coef=args.curr_loss_coef, tc_loss_coef=args.tc_loss_coef, tc_start_epoch=0, tc_curr_coef=args.tc_curr_coef, temperature=0.1)
        elif args.method == 'simsiam':
            self.model = SimSiamSupCon(total_batch_size=args.replay_batch_size, replay_batch_size=args.replay_batch_size, group_norm=args.group_norm, depth=args.depth, curr_loss_coef=args.curr_loss_coef, tc_loss_coef=args.tc_loss_coef, tc_start_epoch=0, tc_curr_coef=args.tc_curr_coef, temperature=0.1)
        else:
            raise ValueError("Unknown Method")

        self.model = self.model.cuda()

        if args.method == 'simsiam':
            parameters = [{'params': list(self.model.backbone.parameters()) + list(self.model.projector.parameters()) + list(self.model.supcon_projector.parameters()), 'fix_lr': False}, {'params': self.model.predictor.parameters(), 'fix_lr': True}]
            self.optimizer = torch.optim.SGD(parameters, lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        os.makedirs(args.save_dir, exist_ok=True)

        self.epoch = 1
        self.model.epoch = 1

    def train(self):
        self.model.train()

        self.longmem_sampler = ReservoirBatchSampler(data_source=self.longmem_dataset, buffer_size=self.args.long_buffer_size, step_frame_size=self.args.curr_batch_size, batch_size=self.args.replay_batch_size, init_buffer_size=self.args.long_buffer_size)

        self.longmem_dataloader = DataLoader(
                self.longmem_dataset,
                batch_sampler=self.longmem_sampler,
                num_workers=self.num_workers,
                pin_memory=True, 
                prefetch_factor=1)
        
        for step, (data_batch, labels) in enumerate(self.longmem_dataloader):
            data_batch = data_batch.cuda()
            labels = labels.cuda()

            loss, log_dict = self.model(data_batch, labels)
            loss.backward()

            if self.global_step % self.args.grad_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.global_step % self.args.log_freq == 0:
                wandb.log(log_dict, step=self.global_step)

            if self.global_step % self.args.print_freq == 0:
                print(f'Time {datetime.now().strftime("%Y/%m/%d %H:%M:%S")} Step {step} Total Step {self.global_step} Loss {loss:.4f}')

            self.global_step += 1
            
            if (self.global_step+1) % self.args.eval_freq == 0 or self.global_step == len(self.longmem_dataloader) - 1:
                self.save_checkpoint()
                if self.args.imagenet_eval:
                    eval_imagenet(self.model, self.num_workers, self.args.crop_size, self.global_step)
                if self.args.labeledS_eval:
                    eval_labeledS(self.model, self.num_workers, self.args.crop_size, self.global_step)
                self.model.train()

    def save_checkpoint(self):
        filepath = osp.join(self.args.save_dir, f'step_{self.global_step+1}.pth')
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict':  self.optimizer.state_dict()}, filepath)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_freq', default=100, type=int, help='log frequency (default: 8)')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency (default: 100)')
    parser.add_argument('--eval_freq', default=10000, type=int, help='eval frequency (default: 10000)')

    parser.add_argument('--crop_size', default=112, type=int, help='Image Crop Size (default: 112)')

    parser.add_argument('--curr_batch_size', default=64, type=int, help='Batch Size of Current Stream (default: 256)')
    parser.add_argument('--replay_batch_size', default=256, type=int, help='Batch Size of Replay Stream (default: 256)')
    parser.add_argument('--long_buffer_size', default=10000, type=int, help='size of long term memory replay buffer')
    parser.add_argument('--depth', default=50, type=int, choices=[18, 50], help='size of replay buffer')
    parser.add_argument('--class_length', default=0, type=int, help='default length of each class')

    parser.add_argument('--grad_accumulation', default=1, type=int, help='how many batches to accumulate grads before making an update')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    parser.add_argument('--curr_loss_coef', default=1.0, type=float, help='coefficient of cross loss')
    parser.add_argument('--tc_loss_coef', default=1.0, type=float, help='coefficient of tc loss')
    parser.add_argument('--tc_curr_coef', default=0.0, type=float, help='coefficient of tc loss on current batch')

    parser.add_argument('--scale_min', default=0.08, type=float, help='min value of scale in data aug')
    parser.add_argument('--scale_max', default=1.0, type=float, help='max value of scale in data aug')

    parser.add_argument('--save_dir', default='results/', type=str, help='dir to save checkpoints')
    parser.add_argument('--imagenet_eval', action='store_true', default=False, help='whether to do imagenet eval')
    parser.add_argument('--labeledS_eval', action='store_true', default=False, help='whether to do labeledS eval')
    parser.add_argument('--group_norm', action='store_true', default=False, help='whether to use group norm instead of batch norm')
    parser.add_argument('--ensure_overlap', action='store_true', default=False, help='whether to ensure overlap in crops')

    # TODO add additional method names here
    parser.add_argument('--method', default='simclr_supcon', type=str, choices=['simclr', 'simsiam', 'osiris', 'tc'], help='which method to use')
    parser.add_argument('--dataset', default='saycam', type=str, choices=['saycam', 'kcam'], help='which dataset to use')

    args = parser.parse_args()

    args.name = args.save_dir.split('/')[-1]
    
    if args.class_length == 0:
        args.class_length = None

    if args.dataset == 'saycam':
        args.dataset_subsample = 1
    else:
        args.dataset_subsample = 1

    wandb.init(project="saycam-ssl", name=args.name, config=args)

    trainer = Trainer(args)
    trainer.train()

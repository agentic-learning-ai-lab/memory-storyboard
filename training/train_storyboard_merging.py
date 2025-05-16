import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
from utils import greedy_hierarchical_clustering
from osiris_model import SimSiamSupCon, SimCLRTC, SimCLRSupCon, OsirisSupcon, SimSiamTC
import wandb
import argparse
from datetime import datetime
from dataset_utils import *
from eval_model import eval_imagenet, eval_labeledS
from replay_buffer import FIFOBatchSampler, ReservoirBatchSampler
from dataset_newsampler import StreamingDataset
import pickle
from PIL import Image

# wandb.require("core")

class Trainer():
    def __init__(self, args):

        self.args = args
        self.global_step = 0

        num_workers = int(len(os.sched_getaffinity(0))) # num CPUs on the machine
        self.num_workers = num_workers

        os.makedirs(args.save_dir, exist_ok=True)

        self.longmem_dataset = StreamingDataset(dataset=args.dataset, scale_min=args.scale_min, scale_max=args.scale_max, ensure_overlap=args.ensure_overlap, subsample=args.dataset_subsample, class_length=args.class_length, crop_size=self.args.crop_size)

        self.shortmem_dataset = StreamingDataset(dataset=args.dataset, scale_min=args.scale_min, scale_max=args.scale_max, ensure_overlap=args.ensure_overlap, subsample=args.dataset_subsample, class_length=args.class_length, crop_size=self.args.crop_size)

        if args.method == 'osiris': # Need to check the paramter values
            self.model = OsirisSupcon(args.curr_batch_size, args.replay_batch_size, group_norm=args.group_norm, curr_loss_coef=0, cross_loss_coef=args.curr_loss_coef, replay_loss_coef=args.curr_loss_coef, tc_loss_coef=0, tc_style='replay', supcon_loss_coef=0, depth=args.depth, sep_replay=False)
        elif args.method == 'tc':
            num_total_classes = len(self.longmem_dataset) // (args.class_length // args.dataset_subsample) + 1
            self.model = SimCLRTC(total_batch_size=args.curr_batch_size + args.replay_batch_size, replay_batch_size=args.replay_batch_size, group_norm=args.group_norm, depth=args.depth, curr_loss_coef=0, tc_loss_coef=args.tc_loss_coef, num_total_classes=num_total_classes, tc_start_epoch=0)
        elif args.method == 'simclr_tc':
            num_total_classes = len(self.longmem_dataset) // (args.class_length // args.dataset_subsample) + 1
            self.model = SimCLRTC(total_batch_size=args.curr_batch_size + args.replay_batch_size, replay_batch_size=args.replay_batch_size, group_norm=args.group_norm, depth=args.depth, curr_loss_coef=args.curr_loss_coef, tc_loss_coef=args.tc_loss_coef, num_total_classes=num_total_classes, tc_start_epoch=0)
        elif args.method == 'simclr':
            self.model = SimCLRSupCon(total_batch_size=args.curr_batch_size + args.replay_batch_size, replay_batch_size=args.replay_batch_size, group_norm=args.group_norm, depth=args.depth, curr_loss_coef=args.curr_loss_coef, tc_loss_coef=args.tc_loss_coef, tc_start_epoch=0, tc_curr_coef=args.tc_curr_coef, temperature=0.1)
        elif args.method == 'simsiam_tc':
            num_total_classes = len(self.longmem_dataset) // (args.class_length // args.dataset_subsample) + 1
            self.model = SimSiamTC(total_batch_size=args.curr_batch_size + args.replay_batch_size, replay_batch_size=args.replay_batch_size, group_norm=args.group_norm, depth=args.depth, curr_loss_coef=args.curr_loss_coef, tc_loss_coef=args.tc_loss_coef, num_total_classes=num_total_classes, tc_start_epoch=0)
        elif args.method == 'simsiam':
            self.model = SimSiamSupCon(total_batch_size=args.curr_batch_size + args.replay_batch_size, replay_batch_size=args.replay_batch_size, group_norm=args.group_norm, depth=args.depth, curr_loss_coef=args.curr_loss_coef, tc_loss_coef=args.tc_loss_coef, tc_start_epoch=0, tc_curr_coef=args.tc_curr_coef, temperature=0.1)
        else:
            raise ValueError("Unknown Method")

        self.model = self.model.cuda()

        if args.method == 'simsiam':
            parameters = [{'params': list(self.model.backbone.parameters()) + list(self.model.projector.parameters()) + list(self.model.supcon_projector.parameters()), 'fix_lr': False}, {'params': self.model.predictor.parameters(), 'fix_lr': True}]
            self.optimizer = torch.optim.SGD(parameters, lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        elif args.method == 'simsiam_tc':
            parameters = [{'params': list(self.model.backbone.parameters()) + list(self.model.projector.parameters()) + list(self.model.classifer.parameters()), 'fix_lr': False}, {'params': self.model.predictor.parameters(), 'fix_lr': True}]
            self.optimizer = torch.optim.SGD(parameters, lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.epoch = 1
        self.model.epoch = 1

        self.shortmem_base_label = 0
        self.prototypes = {}
        self.merged_classes = 0
        self.label_mapping = {}

    def train(self):
        self.model.train()

        self.shortmem_sampler = FIFOBatchSampler(data_source=self.shortmem_dataset, buffer_size=self.args.short_buffer_size, step_frame_size=self.args.step_batch_size, batch_size=self.args.curr_batch_size, subsample=self.args.subsample, init_buffer_size=self.args.long_buffer_size)

        self.longmem_sampler = ReservoirBatchSampler(data_source=self.longmem_dataset, buffer_size=self.args.long_buffer_size, step_frame_size=self.args.step_batch_size, batch_size=self.args.replay_batch_size, init_buffer_size=self.args.long_buffer_size)

        self.shortmem_dataloader = DataLoader(
                self.shortmem_dataset,
                batch_sampler=self.shortmem_sampler,
                num_workers=self.num_workers,
                pin_memory=True, 
                prefetch_factor=1)

        self.longmem_dataloader = DataLoader(
                self.longmem_dataset,
                batch_sampler=self.longmem_sampler,
                num_workers=self.num_workers,
                pin_memory=True, 
                prefetch_factor=1)
        
        assert len(self.shortmem_dataloader) == len(self.longmem_dataloader), print(len(self.shortmem_dataloader), len(self.longmem_dataloader))

        for step, ((curr_batch, curr_labels), (replay_batch, replay_labels)) in enumerate(zip(self.shortmem_dataloader, self.longmem_dataloader)):

            if self.global_step == 0:
                self.init_prototypes()

            data_batch = torch.cat([curr_batch, replay_batch], dim=0)
            labels = torch.cat([curr_labels, replay_labels], dim=0)
            data_batch = data_batch.cuda()
            labels = labels.cuda()

            for _ in range(self.args.num_grads):
                loss, log_dict = self.model(data_batch, labels)
                loss.backward()

                if self.global_step % self.args.grad_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if self.shortmem_sampler.buffer[0]['label'] >= self.shortmem_base_label:
                # Need Reclustering on the Short Mem
                num_clusters = self.shortmem_sampler.buffer[-1]['label'] - self.shortmem_sampler.buffer[0]['label'] + 1
                change_points_frame_id = self.cluster_shortmem(num_clusters=num_clusters)
                # Update the labels in the long term buffer
                longmem_frame_ids = self.longmem_sampler.get_frame_ids(threshold=change_points_frame_id[0])
                longmem_cluster_ids = torch.searchsorted(torch.tensor(change_points_frame_id), torch.tensor(longmem_frame_ids), side='right')
                # import pdb; pdb.set_trace()
                new_longmem_labels = longmem_cluster_ids + self.shortmem_sampler.buffer[0]['label'] - 1
                self.longmem_dataset.set_labels(longmem_frame_ids, new_longmem_labels)
                self.shortmem_base_label = self.shortmem_sampler.buffer[-1]['label']
                self.merge_labels(change_points_frame_id, num_samples_per_class=32, threshold='dynamic')

            if self.global_step % self.args.log_freq == 0:
                wandb.log(log_dict, step=self.global_step)

            if self.global_step % self.args.print_freq == 0:
                print(f'Time {datetime.now().strftime("%Y/%m/%d %H:%M:%S")} Step {step} Total Step {self.global_step} Loss {loss:.4f}')

            self.global_step += self.args.num_grads
            
            if (self.global_step // self.args.num_grads +1) % self.args.eval_freq == 0 or self.global_step // self.args.num_grads == len(self.shortmem_dataloader) - 1:
                self.save_checkpoint()
                if self.args.imagenet_eval:
                    eval_imagenet(self.model, self.num_workers, self.args.crop_size, self.global_step)
                if self.args.labeledS_eval:
                    eval_labeledS(self.model, self.num_workers, self.args.crop_size, self.global_step)
                self.model.train()

        print("Training Finished")
        print("label mapping", self.label_mapping)

    def cluster_shortmem(self, num_clusters):
        self.model.eval()
        print(f"------ Start clustering")
        indices = [b['idx'] for b in self.shortmem_sampler.buffer]
        imgs, frame_ids = self.shortmem_dataset.load_imgs_test(indices)
        base_label = self.shortmem_sampler.buffer[0]['label']
        with torch.no_grad():
            imgs = imgs.cuda()
            all_embds = self.model.backbone(imgs).cpu()

        all_embds = all_embds / all_embds.norm(dim=1, keepdim=True)
        similarity_matrix = torch.mm(all_embds, all_embds.t())

        change_points = greedy_hierarchical_clustering(similarity_matrix, num_clusters=num_clusters)
        print(f"------ Finished clustering, Change Points {change_points}")

        change_points_frame_id = [frame_ids[0]] + [frame_ids[change_point] for change_point in change_points]
        class_labels = torch.searchsorted(torch.tensor(change_points), torch.tensor(range(self.shortmem_sampler.get_buffer_length())), side='right')
        class_labels += base_label
        self.shortmem_dataset.set_labels(indices, class_labels)
        self.model.train()
        return change_points_frame_id

    def init_prototypes(self, num_samples_per_class=32):
        self.model.eval()
        # Randomly sample num_samples_per_class indices for each label
        unique_labels = range(self.shortmem_sampler.buffer[0]['label'])
        # get a mapping from label to indices
        label_to_indices = {label: [] for label in unique_labels}
        for idx, label in enumerate(self.longmem_dataset.labels.tolist()):
            if label in unique_labels:
                label_to_indices[label].append(idx)
        # Randomly sample num_samples_per_class indices for each label
        sampled_indices = []
        for label, indices in label_to_indices.items():
            if len(indices) > num_samples_per_class:
                sampled_indices.append(random.sample(indices, num_samples_per_class))
            else:
                sampled_indices.append(indices)
        for label, indices in enumerate(sampled_indices):
            imgs = []
            for idx in indices:
                img_path, _, _ = self.longmem_dataset.curr_base_fns[idx]
                img = Image.open(img_path)
                img = self.longmem_dataset.test_transforms(img).unsqueeze(0)
                imgs.append(img)
            self.prototypes[label+self.merged_classes] = torch.cat(imgs, dim=0)
        self.merged_classes = len(self.prototypes)
        print("Init Prototype Classes", len(self.prototypes))

    def merge_labels(self, change_points_frame_id, num_samples_per_class=32, threshold='dynamic'):
        print(f"------ Start preparing prototypes")
        self.model.eval()
        # Randomly sample num_samples_per_class indices for each label
        sampled_indices = []
        for idx in range(len(change_points_frame_id)-1):
            indices = range(change_points_frame_id[idx], change_points_frame_id[idx+1])
            if len(indices) > num_samples_per_class:
                sampled_indices.append(indices[::len(indices)//num_samples_per_class][:num_samples_per_class])
            else:
                sampled_indices.append(indices)
        for label, indices in enumerate(sampled_indices):
            imgs = []
            for idx in indices:
                img_path, _, _ = self.longmem_dataset.curr_base_fns[idx]
                img = Image.open(img_path)
                img = self.longmem_dataset.test_transforms(img).unsqueeze(0)
                imgs.append(img)
            self.prototypes[label+self.merged_classes] = torch.cat(imgs, dim=0)
        # Get the average embedding of each class for the sampled indices
        if self.shortmem_base_label > self.args.warmup:
            print(f"------ Start merging classes")
            # Do the merging
            sampled_embeddings = []
            for imgs in self.prototypes.values():
                with torch.no_grad():
                    imgs = imgs.cuda()
                    embedding = trainer.model.backbone(imgs).cpu()
                    imgs = imgs.to('cpu')
                sampled_embeddings.append(torch.mean(embedding, dim=0))
            sampled_embeddings = torch.stack(sampled_embeddings, dim=0)
            sampled_embeddings = torch.nn.functional.normalize(sampled_embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(sampled_embeddings, sampled_embeddings.t())
            # import pdb; pdb.set_trace()
            for j in range(max(self.merged_classes, 1), similarity_matrix.size(0)):
                similarities = similarity_matrix[:j, j]
                max_value, max_index = torch.max(similarities, dim=0)
                if threshold == 'dynamic':
                    threshold = torch.quantile(similarity_matrix, 1 - 1/len(similarity_matrix) - self.args.merge_threshold).item()
                    print(f"Dynamic Threshold {threshold}")
                if max_value > threshold:
                    new_label = self.label_mapping.get(max_index.item(), max_index.item())
                    self.label_mapping[j] = new_label
                    print(f"Merging class {j} into class {new_label}")
                    new_indices = range(change_points_frame_id[j-self.merged_classes], change_points_frame_id[j+1-self.merged_classes])
                    new_labels = torch.ones(len(new_indices)) * new_label
                    trainer.longmem_dataset.set_labels(new_indices, new_labels)
            self.merged_classes = len(self.prototypes)
            assert self.merged_classes == self.shortmem_base_label
        else:
            self.merged_classes = len(self.prototypes)

    def save_checkpoint(self):
        filepath = osp.join(self.args.save_dir, f'step_{self.global_step+1}.pth')
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict':  self.optimizer.state_dict()}, filepath)
        filepath_mapping = osp.join(self.args.save_dir, f'label_mapping.pkl')
        with open(filepath_mapping, 'wb') as f:
            pickle.dump(self.label_mapping, f)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_freq', default=100, type=int, help='log frequency (default: 8)')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency (default: 100)')
    parser.add_argument('--eval_freq', default=10000, type=int, help='eval frequency (default: 10000)')

    parser.add_argument('--crop_size', default=112, type=int, help='Image Crop Size (default: 112)')

    parser.add_argument('--step_batch_size', default=64, type=int, help='Batch Size of Current Stream (default: 256)')
    parser.add_argument('--curr_batch_size', default=64, type=int, help='Batch Size of Current Stream (default: 256)')
    parser.add_argument('--replay_batch_size', default=192, type=int, help='Batch Size of Replay Stream (default: 256)')
    parser.add_argument('--long_buffer_size', default=10000, type=int, help='size of long term memory replay buffer')
    parser.add_argument('--short_buffer_size', default=1000, type=int, help='size of short term memory replay buffer')
    parser.add_argument('--depth', default=50, type=int, choices=[18, 50], help='size of replay buffer')
    parser.add_argument('--class_length', default=0, type=int, help='default length of each class')

    parser.add_argument('--grad_accumulation', default=1, type=int, help='how many batches to accumulate grads before making an update')
    parser.add_argument('--num_grads', default=1, type=int, help='how many gradient update steps to take for each batch')
    parser.add_argument('--subsample', default=1, type=int, help='subsampling rate for the short term buffer')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--warmup', default=0, type=int, help='number of warmup epochs; set warmup to very big number to disable clustering')

    parser.add_argument('--curr_loss_coef', default=1.0, type=float, help='coefficient of cross loss')
    parser.add_argument('--tc_loss_coef', default=1.0, type=float, help='coefficient of tc loss')
    parser.add_argument('--tc_curr_coef', default=0.0, type=float, help='coefficient of tc loss on current batch')

    parser.add_argument('--scale_min', default=0.08, type=float, help='min value of scale in data aug')
    parser.add_argument('--scale_max', default=1.0, type=float, help='max value of scale in data aug')

    parser.add_argument('--merge_threshold', default=0.001, type=float, help='max value of scale in data aug')

    parser.add_argument('--save_dir', default='results/', type=str, help='dir to save checkpoints')
    parser.add_argument('--imagenet_eval', action='store_true', default=False, help='whether to do imagenet eval')
    parser.add_argument('--labeledS_eval', action='store_true', default=False, help='whether to do labeledS eval')
    parser.add_argument('--group_norm', action='store_true', default=False, help='whether to use group norm instead of batch norm')
    parser.add_argument('--ensure_overlap', action='store_true', default=False, help='whether to ensure overlap in crops')

    # TODO add additional method names here
    parser.add_argument('--method', default='simclr_supcon', type=str, choices=['simclr', 'simclr_tc', 'simsiam', 'simsiam_tc', 'osiris', 'tc'], help='which method to use')
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

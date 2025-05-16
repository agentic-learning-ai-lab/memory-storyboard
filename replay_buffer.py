import torch
import random
from collections import deque
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

class Buffer():
    """Buffer updated with reservoir sampling."""
    def __init__(self, max_size, decay_weight=1.0):
        self.max_size = max_size
        self.decay_weight = decay_weight
        self.buffer = []
        self.buffer_weights = torch.zeros(0)

    def update(self, new_data):
        new_data = list(new_data)
        self.buffer_weights *= self.decay_weight
        new_weights = torch.rand(len(new_data))
        cat_weights = torch.cat([new_weights, self.buffer_weights])
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)
        buffer_idxs = sorted_idxs[: self.max_size]
        self.num_new_elements = sum(buffer_idxs < len(new_weights)).item()
        new_data.extend(self.buffer)
        self.buffer = [new_data[i] for i in buffer_idxs]
        self.buffer_weights = sorted_weights[: self.max_size]

    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def get_buffer(self):
        if 'idx' in self.buffer[0]:
            return sorted(self.buffer, key=lambda item: item['idx'])
        else:
            return sorted(self.buffer)

    def __len__(self):
        return len(self.buffer)

class IIDBatchSampler(torch.utils.data.Sampler):
    """Buffer updated with first-in, first-out strategy."""
    def __init__(self, data_source, batch_size, step_frame_size=64):
        self.data_source = data_source
        self.sampler = SequentialSampler(data_source=self.data_source)
        self.batch_size = batch_size
        self.step_frame_size = step_frame_size
        self.num_batches_yielded = 0

    def __iter__(self):
        all_indices = list(self.sampler)
        while self.num_batches_yielded < len(self):
            indices = random.sample(all_indices, k=self.batch_size)
            self.num_batches_yielded += 1
            yield indices

    def __len__(self) -> int:
        return len(self.sampler) // self.step_frame_size
    
class SequentialBatchSampler(torch.utils.data.Sampler):
    """Buffer updated with first-in, first-out strategy."""
    def __init__(self, data_source, batch_size, step_frame_size, subsample=1):

        self.data_source = data_source
        self.sampler = SequentialSampler(data_source=self.data_source)

        self.step_frame_size = step_frame_size
        self.batch_size = batch_size

        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.subsample = subsample

        self.labels = None

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen

    def sample_k(self, q, k):
        if k < len(q):
            return random.sample(q, k=k)
        elif k == len(q):
            return q
        else:
            return random.choices(q, k=k)

    def __iter__(self):
        self.num_batches_seen = 0
        self.num_batches_yielded = 0

        all_indices = list(self.sampler)
        while self.num_batches_yielded < len(self) and self.db_head < len(all_indices):
            indices = all_indices[self.db_head:self.db_head + self.batch_size:self.subsample]
            self.db_head += self.step_frame_size
            self.num_batches_yielded += 1
            yield indices

    def __len__(self) -> int:
        return len(self.sampler) // self.step_frame_size

class FIFOBatchSampler(torch.utils.data.Sampler):
    """Buffer updated with first-in, first-out strategy."""
    def __init__(self, data_source, buffer_size, batch_size, step_frame_size, init_buffer_size, subsample=1):

        self.data_source = data_source
        self.sampler = SequentialSampler(data_source=self.data_source)

        self.buffer_size = buffer_size
        self.step_frame_size = step_frame_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.subsample = subsample

        self.labels = None

        self.init_buffer_size = init_buffer_size

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen

    def sample_k(self, q, k):
        if k < len(q):
            return random.sample(q, k=k)
        elif k == len(q):
            return q
        else:
            return random.choices(q, k=k)

    def __iter__(self):
        self.num_batches_seen = 0
        self.num_batches_yielded = 0

        all_indices = list(self.sampler)
        while self.num_batches_yielded < len(self):
            if self.db_head < len(all_indices):
                indices = all_indices[self.db_head:self.db_head + self.step_frame_size:self.subsample]
                self.buffer += [{'idx': idx, 'label': self.data_source.get_default_label(idx)} for idx in indices]
                self.db_head += len(indices) * self.subsample
                if len(indices) > 0 and self.db_head < self.init_buffer_size:
                    continue
            batch = self.sample_k(self.buffer, self.batch_size)
            batch_idx = [b['idx'] for b in batch]
            batch_labels = [b['label'] for b in batch]
            self.num_batches_yielded += 1
            # print("FIFO Batch", sorted(batch_idx), "Buffer Range", self.buffer[0]['idx'], self.buffer[-1]['idx'])
            # print("FIFO Batch", sorted(batch_labels))
            yield batch_idx

    def __len__(self) -> int:
        return len(self.sampler) // self.step_frame_size

    def get_buffer_length(self):
        return len(self.buffer)
    

class ReservoirBatchSampler(torch.utils.data.Sampler):
    """Buffer updated with first-in, first-out strategy."""
    def __init__(self, data_source, buffer_size, batch_size, step_frame_size, init_buffer_size):

        # self.seed_base = 93823982
        self.data_source = data_source
        # self.generator = Generator()
        # self.generator.manual_seed(self.seed_base)
        # random.seed(self.seed_base)
        self.sampler = SequentialSampler(data_source=self.data_source)

        self.buffer_size = buffer_size
        self.step_frame_size = step_frame_size
        self.batch_size = batch_size
        self.buffer = Buffer(max_size=buffer_size)

        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0

        self.labels = None

        self.init_buffer_size = init_buffer_size

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen

    def sample_k(self, q, k):
        if k < len(q):
            return random.sample(q, k=k)
        elif k == len(q):
            return q
        else:
            return random.choices(q, k=k)

    def __iter__(self):
        self.num_batches_seen = 0
        self.num_batches_yielded = 0

        all_indices = list(self.sampler)
        while self.num_batches_yielded < len(self):
            if self.db_head < len(all_indices):
                indices = all_indices[self.db_head:self.db_head + self.step_frame_size]
                self.buffer.update([{'idx': idx, 'label': self.data_source.get_default_label(idx)} for idx in indices])
                self.db_head += len(indices)
                if len(indices) > 0 and len(self.buffer) < self.init_buffer_size:
                    continue
            batch = self.sample_k(self.buffer.get_buffer(), self.batch_size)
            batch_idx = [b['idx'] for b in batch]
            batch_labels = [b['label'] for b in batch]
            self.num_batches_yielded += 1
            # print("Long Buffer Batch", sorted(batch_idx))
            # print("Long Batch", sorted(batch_labels))
            yield batch_idx

    def __len__(self) -> int:
        return len(self.sampler) // self.step_frame_size

    def get_frame_ids(self, threshold=0):
        return [b['idx'] for b in self.buffer.get_buffer() if b['idx'] > threshold]
    
    def get_buffer_length(self):
        return len(self.buffer)


class MinRedBatchSampler(torch.utils.data.Sampler):
    """Buffer updated with MinRed strategy."""
    def __init__(self, data_source, buffer_size, batch_size, step_frame_size, init_buffer_size, gamma=0.5):

        self.data_source = data_source
        self.sampler = SequentialSampler(data_source=self.data_source)

        self.buffer_size = buffer_size
        self.step_frame_size = step_frame_size
        self.batch_size = batch_size
        self.buffer = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0

        self.labels = None
        self.gamma = gamma
        self.init_buffer_size = init_buffer_size

    def add_to_buffer(self, n):
        if self.db_head >= len(self.all_indices):
            return True

        # Add indices to buffer
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.buffer += [{
                'idx': idx,
                'lifespan': 0,
                'loss': None,
                'similarity': None,
                'neighbor_similarity': None,
                'feature': None,
                'num_seen': 0,
                'seen': False,
            }]
        self.db_head += len(indices_to_add)

        # Increase lifespan count
        for b in self.buffer:
            b['lifespan'] += 1

        return False
    
    def resize_buffer(self, n):
        n2rm = len(self.buffer) - n
        if n2rm <= 0:
            return

        # Only remove samples that have already been evaluated
        buffer = [(b, i) for i, b in enumerate(self.buffer) if b['seen']]
        if len(buffer) < 2 * n2rm:
            lifespans = [b['lifespan'] for b in self.buffer]
            idx2rm = torch.tensor(lifespans).argsort(descending=True)[:n2rm].tolist()
        else:
            # Compute top 5 neighbor average similarity
            feats = torch.stack([b['feature'] for b, i in buffer], 0)
            idx2rm = self.max_coverage_reduction(feats, n2rm)
            # idx2rm = neig_sim.argsort(descending=True)[:n2rm]
            idx2rm = [buffer[i][1] for i in idx2rm]

        # Remove samples from buffer
        idx2rm = set(idx2rm)
        self.buffer = [b for i, b in enumerate(self.buffer) if i not in idx2rm]

        # Recompute nearest neighbor similarity for tracking
        if any(b['seen'] for b in self.buffer):
            feats = torch.stack(
                [b['feature'] for b in self.buffer if b['seen']], 0)
            feats = feats.cuda() if torch.cuda.is_available() else feats
            feats_sim = torch.einsum('ad,bd->ab', feats, feats)
            neig_sim = torch.topk(feats_sim, k=2, dim=-1, sorted=False)[0][:, 1:].mean(dim=1).cpu()
            i = 0
            for b in self.buffer:
                if b['seen']:
                    b['neighbor_similarity'] = neig_sim[i]
                    i += 1

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen

    def update_sample_stats(self, sample_info):
        # sample_info should contain keys 'idx', 'feature1', 'feature2' (detached)
        batch_size = len(sample_info['idx'])
        db2buff = {b['idx']: i for i, b in enumerate(self.buffer)}
        sample_index = sample_info['idx']

        z1, z2 = sample_info['feature1'], sample_info['feature2']
        sample_features = F.normalize(z1 + z2, p=2, dim=-1)

        def polyak_avg(val, avg, gamma):
            return (1 - gamma) * val + gamma * avg

        for i in range(batch_size):
            db_idx = sample_index[i].item()
            if db_idx in db2buff:
                b = self.buffer[db2buff[db_idx]]
                if not b['seen']:
                    b['feature'] = sample_features[i]
                else:
                    b['feature'] = F.normalize(polyak_avg(b['feature'], sample_features[i], self.gamma), p=2, dim=-1)
                b['seen'] = True

    def max_coverage_reduction(self, x, n2rm):
        # removes samples 1 by 1 that are most similar to currently selected.
        sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2
        sim.fill_diagonal_(-10.)
        idx2rm = []
        for i in range(n2rm):
            neig_sim = sim.max(dim=1)[0]
            most_similar_idx = torch.argmax(neig_sim)
            idx2rm += [most_similar_idx.item()]
            sim.index_fill_(0, most_similar_idx, -10.)
            sim.index_fill_(1, most_similar_idx, -10.)
        return idx2rm

    def sample_k(self, q, k):
        if k < len(q):
            return random.sample(q, k=k)
        elif k == len(q):
            return q
        else:
            return random.choices(q, k=k)

    def __iter__(self):
        self.num_batches_seen = 0
        self.num_batches_yielded = 0

        self.all_indices = list(self.sampler)
        while self.num_batches_yielded < len(self):
            done = self.add_to_buffer(self.batch_size)
            if not done and len(self.buffer) < self.buffer_size:
                continue  # keep adding until buffer is full

            self.resize_buffer(self.buffer_size)
            # for j in range(self.repeat):
            batch = self.sample_k(self.buffer, self.batch_size)
            batch_idx = [b['idx'] for b in batch]
            self.num_batches_yielded += 1
            # self.batch_history += [batch_idx]
            yield batch_idx

    def __len__(self) -> int:
        return len(self.sampler) // self.step_frame_size

    def get_frame_ids(self, threshold=0):
        return [b['idx'] for b in self.buffer.get_buffer() if b['idx'] > threshold]
    
    def get_buffer_length(self):
        return len(self.buffer)

class BalancedBatchSampler(torch.utils.data.Sampler):
    """Buffer updated with MinRed strategy."""
    def __init__(self, data_source, buffer_size, batch_size, step_frame_size, init_buffer_size):
        self.data_source = data_source
        self.sampler = SequentialSampler(data_source=self.data_source)
        self.buffer_size = buffer_size
        self.step_frame_size = step_frame_size
        self.batch_size = batch_size
        self.buffer = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.labels = None
        self.init_buffer_size = init_buffer_size
        
    def add_to_buffer(self, n):
        if self.db_head >= len(self.all_indices):
            return True
        # Add indices to buffer
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.buffer += [{
                'idx': idx,
                'label': self.data_source.get_default_label(idx),
            }]
        self.db_head += len(indices_to_add)
        return False
    
    # def balance_buffer(self):
    #     # Count the number of samples in each class
    #     label_counts = {}
    #     for b in self.buffer:
    #         label = b['label']
    #         if label not in label_counts:
    #             label_counts[label] = 1
    #         else:
    #             label_counts[label] += 1
    #     # Iteratively remove samples from the maximum class until buffer size is within limit
    #     while len(self.buffer) > self.buffer_size:
    #         max_label = max(label_counts, key=label_counts.get)
    #         max_class_samples = [b for b in self.buffer if b['label'] == max_label]
    #         sample_to_remove = random.choice(max_class_samples)
    #         self.buffer.remove(sample_to_remove)
    #         label_counts[max_label] -= 1

    def balance_buffer(self):
        # Store samples in a dictionary by class
        samples_by_class = {}
        for b in self.buffer:
            label = b['label']
            if label not in samples_by_class:
                samples_by_class[label] = [b]
            else:
                samples_by_class[label].append(b)
        # Count the number of samples in each class
        label_counts = {label: len(samples) for label, samples in samples_by_class.items()}
        # Iteratively remove samples from the maximum class until buffer size is within limit
        while len(self.buffer) > self.buffer_size:
            max_label = max(label_counts, key=label_counts.get)
            max_class_samples = samples_by_class[max_label]
            sample_to_remove = random.choice(max_class_samples)
            self.buffer.remove(sample_to_remove)
            max_class_samples.remove(sample_to_remove)
            label_counts[max_label] -= 1

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    def sample_k(self, q, k):
        if k < len(q):
            return random.sample(q, k=k)
        elif k == len(q):
            return q
        else:
            return random.choices(q, k=k)
        
    def __iter__(self):
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.all_indices = list(self.sampler)
        while self.num_batches_yielded < len(self):
            done = self.add_to_buffer(self.batch_size)
            if not done and len(self.buffer) < self.buffer_size:
                continue  # keep adding until buffer is full
            self.balance_buffer()
            batch = self.sample_k(self.buffer, self.batch_size)
            batch_idx = [b['idx'] for b in batch]
            self.num_batches_yielded += 1
            yield batch_idx

    def __len__(self) -> int:
        return len(self.sampler) // self.step_frame_size
    
    def get_frame_ids(self, threshold=0):
        return [b['idx'] for b in self.buffer if b['idx'] > threshold]
    
    def get_buffer_length(self):
        return len(self.buffer)
    

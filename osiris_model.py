import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet
import copy
import numpy as np

def _make_projector(sizes):
    """
    make a simple MLP with linear layers followed by ReLU, as in SimCLR
    """
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))

    return nn.Sequential(*layers)

def _make_projector_bn(sizes, bias=True, last_bn=True):
    """
    make a simple MLP with linear layers followed by ReLU, as in SimCLR
    """
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
        layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=bias))
    if last_bn:
        layers.append(nn.BatchNorm1d(sizes[-1]))
        if bias:
            layers[-2].bias.requires_grad = False

    return nn.Sequential(*layers)


def _mask_correlated_samples(batch_size):
    """
    Generate a boolean mask which masks out the similarity between views of the same example in the similarity matrix
    e.g., a mask for batch size = 2 is a 4x4 matrix (due to two augmented views)
        0  1  0  1
        1  0  1  0
        0  1  0  1  
        1  0  1  0 
    """
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask.fill_diagonal_(0)
    mask[:, batch_size:].fill_diagonal_(0)
    mask[batch_size:, :].fill_diagonal_(0)
    return mask


class NT_Xent(nn.Module):
    """
    https://arxiv.org/abs/2002.05709
    Modified from https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
    """
    def __init__(self, batch_size, temperature=0.1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = _mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def forward(self, z_i, z_j):
        """
        Standard contrastive loss on [z_i, z_j]

        param z_i (bsz, d): the stacked g(f(x)) for one augmented view x
        param z_j (bsz, d): the stacked g(f(x')) for the other view x'
        
        returns loss
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        batch_size = z_i.size(0)
        N = 2 * batch_size
        
        z = torch.cat((z_i, z_j), dim=0)
        sim = z @ z.t()

        # positives are the similarity between different views of the same example 
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # negatives are the similarity between different examples
        mask = _mask_correlated_samples(batch_size) if batch_size != self.batch_size else self.mask     # accounts for the last batch of the epoch
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class NT_XentWeighted(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super(NT_XentWeighted, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._mask_correlated_samples(batch_size)

    def _mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, correlation_matrix):
        batch_size = z_i.size(0)
        N = 2 * batch_size

        # Normalize embeddings
        z = torch.cat([F.normalize(z_i, dim=1), F.normalize(z_j, dim=1)], dim=0)

        # Similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature

        # Mask to exclude self-similarity and positive pairs
        mask = self.mask.to(z.device)
        correlation_matrix = correlation_matrix.to(z.device)
        correlation_matrix = correlation_matrix * mask.float()

        # Numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exponentials
        exp_sim = torch.exp(sim) * mask.float()

        # Weighted numerator
        numerator = torch.sum(exp_sim * correlation_matrix, dim=1)

        # Denominator
        denominator = torch.sum(exp_sim, dim=1)

        # Log probability
        log_prob = torch.log(numerator / (denominator + 1e-8))

        # Loss
        loss = -torch.mean(log_prob)

        return loss
    

class Cross_NT_Xent(nn.Module):
    """
    Cross-task loss in Osiris
    """
    def __init__(self, batch_size, temperature=0.1):
        super(Cross_NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j, u_i, u_j):
        """
        Contrastive loss for discriminating z and u
        No comparison between examples within z or u

        param z_i (bsz, d): the stacked h(f(x)) for one augmented view x from the current task
        param z_j (bsz, d): the stacked h(f(x')) for the other view x' from the current task
        param u_i (p*bsz, d): the stacked h(f(y)) for one augmented view y from the memory
        param u_j (p*bsz, d): the stacked h(f(y')) for the other view y' from the memory
        
        returns loss
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        u_i = F.normalize(u_i, p=2, dim=1)
        u_j = F.normalize(u_j, p=2, dim=1)

        batch_size = z_i.size(0)
        N = batch_size * 2
        # positives are the similarity between different views of the same example within z
        positive_samples = torch.sum(z_i*z_j, dim=-1).repeat(2).reshape(N, 1)

        # negatives are comparisons between the examples in z and the ones in u
        z = torch.cat([z_i, z_j], dim=0)
        u = torch.cat([u_i, u_j], dim=0)
        negative_samples = z @ u.t()

        # loss
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat([positive_samples, negative_samples], dim=1) / self.temperature
        loss_zu = self.criterion(logits, labels)
        loss_zu /= N
        
        # for a symmetric loss, switch z and u
        # we do not need to recompute the similarity matrix between z and u
        # simply use the columns rather than the rows of the matrix as negatives
        batch_size = u_i.size(0)
        N = batch_size * 2
        positive_samples = torch.sum(u_i*u_j, dim=-1).repeat(2).reshape(N, 1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat([positive_samples, negative_samples.t()], dim=1) / self.temperature
        loss_uz = self.criterion(logits, labels)
        loss_uz /= N

        # final cross-task loss
        loss = 0.5 * (loss_zu + loss_uz)

        return loss

class SupCon(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        super(SupCon, self).__init__()
        self.temperature = temperature

    def forward(self, feats, labels):
        device = torch.device('cuda')
        feats = F.normalize(feats, dim=-1, p=2)
        mask = torch.eq(labels.view(-1, 1), labels.contiguous().view(1, -1)).float().to(device)
        self.logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(mask.shape[0]).view(-1, 1).to(device), 0)
        self.mask = mask * self.logits_mask
        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        logits = F.log_softmax(logits, dim=-1)
        loss = torch.sum(p * logits, dim=-1)

        return -loss.mean()

class TemporalClassification(nn.Module):
    def __init__(self, batch_size, filter_unseen=True, filter_classes=None, num_total_classes=2000):
        super(TemporalClassification, self).__init__()
        self.batch_size = batch_size
        self.num_total_classes = num_total_classes
        self.filter_classes = filter_classes
        self.filter_unseen = filter_unseen
        self.max_label = 0

    def forward(self, logits1, logits2, labels):
        N = 2 * self.batch_size
        ce_weight = torch.ones(self.num_total_classes)
        if self.filter_unseen:
            max_label = max(labels)
            if max_label > self.max_label:
                self.max_label = max_label
            ce_weight[self.max_label+1:] = 0
        if self.filter_classes is not None:
            ce_weight[self.filter_classes] = 0
        ce_weight = ce_weight.cuda()
        loss1 = F.cross_entropy(logits1, labels, weight=ce_weight, reduction="sum")
        loss2 = F.cross_entropy(logits2, labels, weight=ce_weight, reduction="sum")
        loss = (loss1 + loss2) / N
        return loss

class SimCLR(nn.Module):
    def __init__(self, total_batch_size, group_norm=False, depth=50):
        super().__init__()
        self.group_norm = group_norm

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        self.criterion_curr = NT_Xent(total_batch_size)

        if depth == 18:
            sizes = [512, 2048, 128]
        elif depth == 50:
            sizes = [2048, 2048, 128]

        self.projector = _make_projector(sizes)

    def forward(self, data_batch, return_features=False):
        x1 = data_batch[:, 0]
        x2 = data_batch[:, 1]
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        loss = self.criterion_curr(z1, z2)
        log_dict = {'curr_loss': loss.item()}

        if return_features:
            return loss, log_dict, f1.detach(), f2.detach()
        else:
            return loss, log_dict

class SimSiam(nn.Module):
    def __init__(self, group_norm=True, depth=50):
        super().__init__()
        self.group_norm = group_norm

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        if depth == 18:
            feature_dim = 512
        elif depth == 50:
            feature_dim = 2048
        
        sizes = [feature_dim, feature_dim, feature_dim, feature_dim]

        self.projector = _make_projector_bn(sizes, bias=False)

        # Predictor network (simple MLP with one hidden layer, as used in SimSiam)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4, bias=False),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim)
        )
        
        # Cosine similarity as a loss
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, data_batch, return_features=False):
        x1 = data_batch[:, 0]
        x2 = data_batch[:, 1]
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        z_i = self.projector(f1)
        z_j = self.projector(f2)

        p_i = self.predictor(z_i)
        p_j = self.predictor(z_j)

        loss_i_j = -self.cosine_similarity(p_i, z_j.detach()).mean()
        loss_j_i = -self.cosine_similarity(p_j, z_i.detach()).mean()

        loss = 0.5 * (loss_i_j + loss_j_i)

        log_dict = {'curr_loss': loss.item()}

        if return_features:
            return loss, log_dict, z_i.detach(), z_j.detach()
        else:
            return loss, log_dict

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class SimCLRTC(nn.Module):
    def __init__(self, total_batch_size, replay_batch_size, group_norm=False, depth=50, curr_loss_coef=1.0, tc_loss_coef=1.0, num_total_classes=2000, tc_start_epoch=0):
        super().__init__()
        self.group_norm = group_norm
        self.curr_loss_coef = curr_loss_coef
        self.tc_loss_coef = tc_loss_coef
        self.total_batch_size = total_batch_size
        self.replay_batch_size = replay_batch_size
        self.tc_start_epoch = tc_start_epoch

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        self.criterion_curr = NT_Xent(total_batch_size)
        self.criterion_tc = TemporalClassification(replay_batch_size, filter_classes=None, num_total_classes=num_total_classes)

        if depth == 18:
            sizes = [512, 2048, 128]
            classifier_sizes = [512, 2048, num_total_classes]
        elif depth == 50:
            sizes = [2048, 2048, 128]
            # sizes = [2048, 512, 2048]
            classifier_sizes = [2048, 2048, num_total_classes]

        self.projector = _make_projector(sizes)
        self.classifer = _make_projector(classifier_sizes)

    def forward(self, data_batch, labels, return_features=False):
        x1 = data_batch[:, 0]
        x2 = data_batch[:, 1]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        if self.curr_loss_coef:
            p1 = self.projector(z1)
            p2 = self.projector(z2)
            loss1 = self.criterion_curr(p1, p2)
        else:
            loss1 = torch.tensor(0)
        if self.tc_loss_coef and self.epoch > self.tc_start_epoch:
            logits1 = self.classifer(z1)[-self.replay_batch_size:]
            logits2 = self.classifer(z2)[-self.replay_batch_size:]
            class_labels = labels[-self.replay_batch_size:]
            # import pdb; pdb.set_trace()
            loss2 = self.criterion_tc(logits1, logits2, class_labels)
        else:
            loss2 = torch.tensor(0)

        log_dict = {'curr_loss': loss1.item(), 'tc_loss': loss2.item()}
        loss = self.curr_loss_coef * loss1 + self.tc_loss_coef * loss2

        if return_features:
            return loss, log_dict, z1.detach(), z2.detach()
        else:
            return loss, log_dict

class SimCLRSupCon(nn.Module):
    def __init__(self, total_batch_size, replay_batch_size, group_norm=False, depth=50, curr_loss_coef=1.0, tc_loss_coef=1.0, tc_curr_coef=0.0, tc_start_epoch=0, fc_eval_numclasses=None, temperature=0.1):
        super().__init__()
        self.group_norm = group_norm
        self.curr_loss_coef = curr_loss_coef
        self.tc_loss_coef = tc_loss_coef
        self.tc_curr_coef = tc_curr_coef
        self.total_batch_size = total_batch_size
        self.replay_batch_size = replay_batch_size
        self.tc_start_epoch = tc_start_epoch

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        self.criterion_curr = NT_Xent(total_batch_size)
        self.criterion_tc = SupCon(temperature=temperature)

        if depth == 18:
            feature_dim = 512
            sizes = [512, 2048, 128]
        elif depth == 50:
            feature_dim = 2048
            sizes = [2048, 2048, 128]
            # sizes = [2048, 512, 2048]

        self.projector = _make_projector(sizes)
        self.supcon_projector = _make_projector(sizes)

        if fc_eval_numclasses:
            self.fc = nn.Linear(in_features=feature_dim, out_features=fc_eval_numclasses, bias=True).cuda()
    
    def eval_forward(self, data_batch):
        z = self.backbone(data_batch)
        return self.fc(z)

    def forward(self, data_batch, labels, return_features=False):
        x1 = data_batch[:, 0]
        x2 = data_batch[:, 1]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        if self.curr_loss_coef:
            p1 = self.projector(z1)
            p2 = self.projector(z2)
            loss1 = self.criterion_curr(p1, p2)
        else:
            loss1 = torch.tensor(0)
        if self.tc_loss_coef and self.epoch > self.tc_start_epoch:
            feats1 = self.supcon_projector(z1)[-self.replay_batch_size:]
            feats2 = self.supcon_projector(z2)[-self.replay_batch_size:]
            feats = torch.cat([feats1, feats2], dim=0)
            class_labels = labels[-self.replay_batch_size:]
            class_labels = torch.cat([class_labels, class_labels], dim=0)
            loss2 = self.criterion_tc(feats, class_labels)
        else:
            loss2 = torch.tensor(0)

        if self.tc_curr_coef and self.epoch > self.tc_start_epoch:
            feats1 = self.supcon_projector(z1)[:-self.replay_batch_size]
            feats2 = self.supcon_projector(z2)[:-self.replay_batch_size]
            feats = torch.cat([feats1, feats2], dim=0)
            class_labels = labels[:-self.replay_batch_size]
            class_labels = torch.cat([class_labels, class_labels], dim=0)
            loss3 = self.criterion_tc(feats, class_labels)
        else:
            loss3 = torch.tensor(0)

        log_dict = {'curr_loss': loss1.item(), 'tc_loss': loss2.item(), 'tc_curr_loss': loss3.item()}
        loss = self.curr_loss_coef * loss1 + self.tc_loss_coef * loss2 + self.tc_curr_coef * loss3

        if return_features:
            return loss, log_dict, z1.detach(), z2.detach()
        else:
            return loss, log_dict
        
class SimSiamTC(nn.Module):
    def __init__(self, total_batch_size, replay_batch_size, group_norm=False, depth=50, curr_loss_coef=1.0, tc_loss_coef=1.0, num_total_classes=2000, tc_start_epoch=0):
        super().__init__()
        self.group_norm = group_norm
        self.curr_loss_coef = curr_loss_coef
        self.tc_loss_coef = tc_loss_coef
        self.total_batch_size = total_batch_size
        self.replay_batch_size = replay_batch_size
        self.tc_start_epoch = tc_start_epoch

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        self.criterion_tc = TemporalClassification(replay_batch_size, filter_classes=None, num_total_classes=num_total_classes)

        if depth == 18:
            feature_dim = 512
            sizes = [512, 2048, 128]
            classifier_sizes = [512, 2048, num_total_classes]
        elif depth == 50:
            feature_dim = 2048
            sizes = [2048, 2048, 128]
            classifier_sizes = [2048, 2048, num_total_classes]

        simsiam_sizes = [feature_dim, feature_dim, feature_dim, feature_dim]

        self.projector = _make_projector_bn(simsiam_sizes, bias=False)
        self.classifer = _make_projector(classifier_sizes)

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4, bias=False),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim)
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, data_batch, labels, return_features=False):
        x1 = data_batch[:, 0]
        x2 = data_batch[:, 1]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        if self.curr_loss_coef:
            z_i = self.projector(z1)
            z_j = self.projector(z2)
            p_i = self.predictor(z_i)
            p_j = self.predictor(z_j)
            loss_i_j = -self.cosine_similarity(p_i, z_j.detach()).mean()
            loss_j_i = -self.cosine_similarity(p_j, z_i.detach()).mean()
            loss1 = 0.5 * (loss_i_j + loss_j_i)
        else:
            loss1 = torch.tensor(0)

        if self.tc_loss_coef and self.epoch > self.tc_start_epoch:
            feats1 = self.classifer(z1)[-self.replay_batch_size:]
            feats2 = self.classifer(z2)[-self.replay_batch_size:]
            class_labels = labels[-self.replay_batch_size:]
            loss2 = self.criterion_tc(feats1, feats2, class_labels)
        else:
            loss2 = torch.tensor(0)

        log_dict = {'curr_loss': loss1.item(), 'tc_loss': loss2.item(),}
        loss = self.curr_loss_coef * loss1 + self.tc_loss_coef * loss2

        if return_features:
            return loss, log_dict, z1.detach(), z2.detach()
        else:
            return loss, log_dict

class SimSiamSupCon(nn.Module):
    def __init__(self, total_batch_size, replay_batch_size, group_norm=False, depth=50, curr_loss_coef=1.0, tc_loss_coef=1.0, tc_curr_coef=0.0, tc_start_epoch=0, fc_eval_numclasses=None, temperature=0.1):
        super().__init__()
        self.group_norm = group_norm
        self.curr_loss_coef = curr_loss_coef
        self.tc_loss_coef = tc_loss_coef
        self.tc_curr_coef = tc_curr_coef
        self.total_batch_size = total_batch_size
        self.replay_batch_size = replay_batch_size
        self.tc_start_epoch = tc_start_epoch

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        self.criterion_tc = SupCon(temperature=temperature)

        if depth == 18:
            feature_dim = 512
            sizes = [512, 2048, 128]
        elif depth == 50:
            feature_dim = 2048
            sizes = [2048, 2048, 128]

        simsiam_sizes = [feature_dim, feature_dim, feature_dim, feature_dim]

        self.projector = _make_projector_bn(simsiam_sizes, bias=False)
        self.supcon_projector = _make_projector(sizes)

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4, bias=False),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim)
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=1)

        if fc_eval_numclasses:
            self.fc = nn.Linear(in_features=feature_dim, out_features=fc_eval_numclasses, bias=True).cuda()
    
    def eval_forward(self, data_batch):
        z = self.backbone(data_batch)
        return self.fc(z)

    def forward(self, data_batch, labels, return_features=False):
        x1 = data_batch[:, 0]
        x2 = data_batch[:, 1]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        if self.curr_loss_coef:
            z_i = self.projector(z1)
            z_j = self.projector(z2)
            p_i = self.predictor(z_i)
            p_j = self.predictor(z_j)
            loss_i_j = -self.cosine_similarity(p_i, z_j.detach()).mean()
            loss_j_i = -self.cosine_similarity(p_j, z_i.detach()).mean()
            loss1 = 0.5 * (loss_i_j + loss_j_i)
        else:
            loss1 = torch.tensor(0)
        if self.tc_loss_coef and self.epoch > self.tc_start_epoch:
            feats1 = self.supcon_projector(z1)[-self.replay_batch_size:]
            feats2 = self.supcon_projector(z2)[-self.replay_batch_size:]
            feats = torch.cat([feats1, feats2], dim=0)
            class_labels = labels[-self.replay_batch_size:]
            class_labels = torch.cat([class_labels, class_labels], dim=0)
            loss2 = self.criterion_tc(feats, class_labels)
        else:
            loss2 = torch.tensor(0)

        if self.tc_curr_coef and self.epoch > self.tc_start_epoch:
            feats1 = self.supcon_projector(z1)[:-self.replay_batch_size]
            feats2 = self.supcon_projector(z2)[:-self.replay_batch_size]
            feats = torch.cat([feats1, feats2], dim=0)
            class_labels = labels[:-self.replay_batch_size]
            class_labels = torch.cat([class_labels, class_labels], dim=0)
            loss3 = self.criterion_tc(feats, class_labels)
        else:
            loss3 = torch.tensor(0)

        log_dict = {'curr_loss': loss1.item(), 'tc_loss': loss2.item(), 'tc_curr_loss': loss3.item()}
        loss = self.curr_loss_coef * loss1 + self.tc_loss_coef * loss2 + self.tc_curr_coef * loss3

        if return_features:
            return loss, log_dict, z1.detach(), z2.detach()
        else:
            return loss, log_dict

class SimCLRSupConAll(nn.Module):
    def __init__(self, total_batch_size, replay_batch_size, group_norm=False, depth=50, curr_loss_coef=1.0, tc_loss_coef=1.0, tc_start_epoch=0):
        super().__init__()
        self.group_norm = group_norm
        self.curr_loss_coef = curr_loss_coef
        self.tc_loss_coef = tc_loss_coef
        self.total_batch_size = total_batch_size
        self.replay_batch_size = replay_batch_size
        self.tc_start_epoch = tc_start_epoch

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        self.criterion_curr = NT_Xent(total_batch_size)
        self.criterion_tc = SupCon()

        if depth == 18:
            sizes = [512, 2048, 128]
        elif depth == 50:
            sizes = [2048, 2048, 128]
            # sizes = [2048, 512, 2048]

        self.projector = _make_projector(sizes)
        self.supcon_projector = _make_projector(sizes)

    def forward(self, data_batch, labels):
        x1 = data_batch[:, 0]
        x2 = data_batch[:, 1]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        p2 = self.projector(z2)
        if self.curr_loss_coef:
            p1 = self.projector(z1)
            p2 = self.projector(z2)
            loss1 = self.criterion_curr(p1, p2)
        else:
            loss1 = torch.tensor(0)
        if self.tc_loss_coef and self.epoch > self.tc_start_epoch:
            feats1 = self.supcon_projector(z1)
            feats2 = self.supcon_projector(z2)
            feats = torch.cat([feats1, feats2], dim=0)
            class_labels = labels
            class_labels = torch.cat([class_labels, class_labels], dim=0)
            loss2 = self.criterion_tc(feats, class_labels)
        else:
            loss2 = torch.tensor(0)

        log_dict = {'curr_loss': loss1.item(), 'tc_loss': loss2.item()}
        loss = self.curr_loss_coef * loss1 + self.tc_loss_coef * loss2
        return loss, log_dict

class Osiris(nn.Module):
    """
    https://arxiv.org/abs/2404.19132
    """
    def __init__(self, curr_batch_size, replay_batch_size, group_norm=False, cross_loss_coef=0.5, replay_loss_coef=0.5, depth=50):
        super().__init__()
        self.group_norm = group_norm
        self.curr_batch_size = curr_batch_size
        self.replay_batch_size = replay_batch_size

        self.cross_loss_coef = cross_loss_coef
        self.replay_loss_coef = replay_loss_coef

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        self.criterion_curr = NT_Xent(curr_batch_size)
        self.criterion_cross = Cross_NT_Xent(curr_batch_size)
        self.criterion_replay = NT_Xent(replay_batch_size)

        if depth == 18:
            sizes = [512, 2048, 128]
        elif depth == 50:
            sizes = [2048, 2048, 128]

        # projector g
        self.projector = _make_projector(sizes)

        # predictor h
        self.predictor = _make_projector(sizes)

        for param in self.predictor.parameters():
            param.requires_grad = False

        self.predictor.load_state_dict(self.projector.state_dict())
        
        for param in self.predictor.parameters():
            param.requires_grad = True

    def forward(self, data_batch):
        """
        param xy1 (bsz+p*bsz, ...): stacked augmented images from the current task (X) and the memory (Y)
        param xy2 (bsz+p*bsz, ...): same images with another sample of augmentations, i.e., X' U Y'
        param mem_idx (bsz+p*bsz): boolean index mask for xy1 and xy2 which gives the index of memory examples, for convenience only
        returns loss
        """

        xy1 = data_batch[:, 0]
        xy2 = data_batch[:, 1]
        
        zu1 = self.backbone(xy1)
        zu2 = self.backbone(xy2)

        z1, z2 = zu1[:-self.replay_batch_size], zu2[:-self.replay_batch_size]
        u1, u2 = zu1[-self.replay_batch_size:], zu2[-self.replay_batch_size:]

        # current task loss
        # on space 1 (i.e., with g o f)
        z1_s1 = self.projector(z1)
        z2_s1 = self.projector(z2)
        loss1 = self.criterion_curr(z1_s1, z2_s1)

        # cross-task loss
        # on space 2 (i.e., with h o f)
        z1_s2 = self.predictor(z1)
        z2_s2 = self.predictor(z2)
        u1_s2 = self.predictor(u1)
        u2_s2 = self.predictor(u2)

        if self.cross_loss_coef:
            loss2 = self.criterion_cross(z1_s2, z2_s2, u1_s2, u2_s2)
        else:
            loss2 = torch.tensor(0)

        # past-task loss
        # also on space 2 (i.e., with h o f)
        if self.replay_loss_coef:
            loss3 = self.criterion_replay(u1_s2, u2_s2)
        else:
            loss3 = torch.tensor(0)

        loss = loss1 + self.cross_loss_coef * loss2 + self.replay_loss_coef * loss3
        log_dict = {'curr_loss': loss1.item(), 'cross_loss': loss2.item(), 'replay_loss': loss3.item()}

        return loss, log_dict

class OsirisSupcon(nn.Module):
    def __init__(self, curr_batch_size, replay_batch_size, group_norm=False, curr_loss_coef=1.0, cross_loss_coef=0.5, replay_loss_coef=0.5, tc_loss_coef=0.5, supcon_loss_coef=0.5, depth=50, tc_style='all', filter_threshold=None, sep_replay=True):
        super().__init__()
        self.group_norm = group_norm
        self.sep_replay = sep_replay

        self.curr_loss_coef = curr_loss_coef
        self.cross_loss_coef = cross_loss_coef
        self.replay_loss_coef = replay_loss_coef
        self.tc_loss_coef = tc_loss_coef
        self.supcon_loss_coef = supcon_loss_coef

        self.curr_batch_size = curr_batch_size
        self.replay_batch_size = replay_batch_size

        if group_norm:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='GN', num_groups=32), activation='mish')
        else:
            self.backbone = ResNet(depth=depth, in_channels=3, norm_cfg=dict(type='SyncBN'), activation='relu')

        self.criterion_curr = NT_Xent(curr_batch_size)
        self.criterion_cross = Cross_NT_Xent(curr_batch_size)
        self.criterion_replay = NT_Xent(replay_batch_size)
        
        self.tc_style = tc_style

        if filter_threshold is None:
            filter_classes = None
        else:
            all_std1 = np.load("std1.npy")
            filter_classes = torch.tensor(all_std1 > filter_threshold, dtype=torch.long)

        if self.tc_style == 'curr':
            self.criterion_tc = TemporalClassification(curr_batch_size, filter_classes=filter_classes)
        elif self.tc_style == 'replay':
            self.criterion_tc = TemporalClassification(replay_batch_size, filter_classes=filter_classes)
        elif self.tc_style == 'all':
            self.criterion_tc = TemporalClassification(curr_batch_size + replay_batch_size, filter_classes=filter_classes)
        else:
            raise ValueError
        
        self.criterion_supcon = SupCon()

        if depth == 18:
            sizes = [512, 2048, 128]
            classifier_sizes = [512, 2048, 2000]
        elif depth == 50:
            sizes = [2048, 2048, 128]
            # sizes = [2048, 512, 2048]
            classifier_sizes = [2048, 2048, 2000]

        # projector g
        self.projector = _make_projector(sizes)

        if self.tc_loss_coef > 0:
            self.classifer = _make_projector(classifier_sizes)
        if self.supcon_loss_coef > 0:
            self.supcon_projector = _make_projector(sizes)

        if self.sep_replay:
            # predictor h
            self.predictor = _make_projector(sizes)
            for param in self.predictor.parameters():
                param.requires_grad = False

            # initialize h from g
            self.predictor.load_state_dict(self.projector.state_dict())
            
            for param in self.predictor.parameters():
                param.requires_grad = True

    def forward(self, data_batch, labels):
        """
        param xy1 (bsz+p*bsz, ...): stacked augmented images from the current task (X) and the memory (Y)
        param xy2 (bsz+p*bsz, ...): same images with another sample of augmentations, i.e., X' U Y'
        
        returns loss
        """

        xy1 = data_batch[:, 0]
        xy2 = data_batch[:, 1]

        zu1 = self.backbone(xy1)
        zu2 = self.backbone(xy2)

        z1, z2 = zu1[:-self.replay_batch_size], zu2[:-self.replay_batch_size]
        u1, u2 = zu1[-self.replay_batch_size:], zu2[-self.replay_batch_size:]

        # current task loss
        # on space 1 (i.e., with g o f)
        z1_s1 = self.projector(z1)
        z2_s1 = self.projector(z2)
        u1_s1 = self.projector(u1)
        u2_s1 = self.projector(u2)
        
        if self.curr_loss_coef:
            loss1 = self.criterion_curr(z1_s1, z2_s1)
        else:
            loss1 = torch.tensor(0)

        if self.sep_replay:
            # cross-task loss
            # on space 2 (i.e., with h o f)
            z1_s2 = self.predictor(z1)
            z2_s2 = self.predictor(z2)
            u1_s2 = self.predictor(u1)
            u2_s2 = self.predictor(u2)

        if self.cross_loss_coef:
            if self.sep_replay:
                loss2 = self.criterion_cross(z1_s2, z2_s2, u1_s2, u2_s2)
            else:
                loss2 = self.criterion_cross(z1_s1, z2_s1, u1_s1, u2_s1)
        else:
            loss2 = torch.tensor(0)

        # past-task loss
        # also on space 2 (i.e., with h o f)
        if self.replay_loss_coef:
            if self.sep_replay:
                loss3 = self.criterion_replay(u1_s2, u2_s2)
            else:
                loss3 = self.criterion_replay(u1_s1, u2_s1)
        else:
            loss3 = torch.tensor(0)

        if self.tc_loss_coef:
            curr_labels, replay_labels = labels[:-self.replay_batch_size], labels[-self.replay_batch_size:]

            if self.tc_style == 'curr':
                logits1 = self.classifer(z1)
                logits2 = self.classifer(z2)
                cls_labels = curr_labels
            elif self.tc_style == 'replay':
                logits1 = self.classifer(u1)
                logits2 = self.classifer(u2)
                cls_labels = replay_labels
            elif self.tc_style == 'all':
                logits1 = self.classifer(zu1)
                logits2 = self.classifer(zu2)
                cls_labels = labels
            else:
                raise ValueError

            loss4 = self.criterion_tc(logits1, logits2, cls_labels)
        else:
            loss4 = torch.tensor(0)

        # TODO: Change this code snippet according to sep_replay
        if self.supcon_loss_coef:
            curr_labels, replay_labels = labels[:-self.replay_batch_size], labels[-self.replay_batch_size:]

            if self.tc_style == 'curr':
                logits1 = self.supcon_projector(z1)
                logits2 = self.supcon_projector(z2)
                feats = torch.cat((logits1, logits2), dim=0)
                cls_labels = torch.cat((curr_labels, curr_labels), dim=0)
            elif self.tc_style == 'replay':
                logits1 = self.supcon_projector(u1)
                logits2 = self.supcon_projector(u2)
                feats = torch.cat((logits1, logits2), dim=0)
                cls_labels = torch.cat((replay_labels, replay_labels), dim=0)
            elif self.tc_style == 'all':
                logits1 = self.supcon_projector(zu1)
                logits2 = self.supcon_projector(zu2)
                feats = torch.cat((logits1, logits2), dim=0)
                cls_labels = torch.cat((labels, labels), dim=0)
            else:
                raise ValueError

            loss5 = self.criterion_supcon(feats, cls_labels)
        else:
            loss5 = torch.tensor(0)

        loss = self.curr_loss_coef * loss1 + self.cross_loss_coef * loss2 + self.replay_loss_coef * loss3 + self.tc_loss_coef * loss4 + self.supcon_loss_coef * loss5
        log_dict = {'curr_loss': loss1.item(), 'cross_loss': loss2.item(), 'replay_loss': loss3.item(), 'tc_loss': loss4.item(), 'supcon_loss': loss5.item()}

        return loss, log_dict

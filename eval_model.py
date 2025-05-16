import torch
import wandb
from datetime import datetime
from evaluate import get_svm_perf, get_knn_perf, get_imagenet_dataloader, get_labeledS_dataloader
from dataset_utils import *

def eval_imagenet(model, num_workers, crop_size, global_step, avg_pool=True):
    model.eval()
    imagenet_dataloader = get_imagenet_dataloader(num_workers=num_workers, image_size=crop_size)
    all_embds = []
    all_labels = []
    with torch.no_grad():
        for step, (eval_images, eval_labels) in enumerate(imagenet_dataloader):
            if (step + 1) % 50 == 0:
                print(f'Time {datetime.now().strftime("%Y/%m/%d %H:%M:%S")} Loading Batch {step + 1}/{len(imagenet_dataloader)}') 

            eval_images = eval_images.cuda()
            if avg_pool:
                res = model.backbone(eval_images, pre_avg_pool=True)
                assert len(res.shape) == 4
                res = torch.mean(res, dim=(2, 3))
            else:
                res = model.encoder(eval_images)
            all_embds.append(res)
            all_labels.append(eval_labels)
    all_embds = torch.cat(all_embds, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0)

    print("Starting ImageNet Eval")
    train_features, test_features = all_embds[:20000], all_embds[20000:]
    train_labels, test_labels = all_labels[:20000], all_labels[20000:]
    acc = get_svm_perf(train_features, test_features, train_labels, test_labels)
    best_k, knn_acc = get_knn_perf(train_features, test_features, train_labels, test_labels)
    print("best_k", best_k)
    wandb.log({'imagenet_acc': acc, 'imagenet_knn_acc': knn_acc}, step=global_step)

    return acc

def eval_labeledS(model, num_workers, crop_size, global_step, avg_pool=True):
    model.eval()
    labeledS_dataloader = get_labeledS_dataloader(num_workers=num_workers, image_size=crop_size)
    all_embds = []
    all_labels = []
    with torch.no_grad():
        for step, (eval_images, eval_labels) in enumerate(labeledS_dataloader):
            eval_images = eval_images.cuda()
            if avg_pool:
                res = model.backbone(eval_images, pre_avg_pool=True)
                assert len(res.shape) == 4
                res = torch.mean(res, dim=(2, 3))
            else:
                res = model.encoder(eval_images)
            all_embds.append(res)
            all_labels.append(eval_labels)
    all_embds = torch.cat(all_embds, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0)

    print("Starting Labeled S Eval")
    train_features, test_features = all_embds[:2886], all_embds[2886:]
    train_labels, test_labels = all_labels[:2886], all_labels[2886:]
    acc = get_svm_perf(train_features, test_features, train_labels, test_labels)
    best_k, knn_acc = get_knn_perf(train_features, test_features, train_labels, test_labels)
    print("best_k", best_k)
    wandb.log({'labeledS_acc': acc, 'labeledS_knn_acc': knn_acc}, step=global_step)
    return acc


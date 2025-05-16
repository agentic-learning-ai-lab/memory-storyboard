import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from dataset_utils import *

class StreamingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset='saycam', scale_min=0.08, scale_max=1.0, ensure_overlap=False, subsample=1, class_length=None, crop_size=112, return_indices=False, label_path=None):
        self.dataset = dataset
        self.ensure_overlap = ensure_overlap
        self.subsample = subsample
        self.class_length = class_length
        self.return_indices = return_indices
        self.label_path = label_path

        if dataset == 'saycam':
            self.num_epochs = 100
        elif dataset == 'kcam':
            self.num_epochs = 164
            
        if self.ensure_overlap:
            transform_crop = RandomResizedCropWithOverlap(scale=(scale_min, scale_max), ratio=(3./4., 4./3.))
        else:
            transform_crop = transforms.RandomResizedCrop(size=(crop_size, crop_size), scale=(scale_min, scale_max), ratio=(0.75, 1.3333))

        transforms_list = [
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
			transforms.RandomGrayscale(p=0.2),
			GaussianBlur(p=0.5), 
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
        if dataset == 'saycam':
            transforms_list.insert(2, transforms.RandomVerticalFlip(p=1.0))
        transform_after_crop = transforms.Compose(transforms_list)

        if self.ensure_overlap:
            def apply_transforms_to_crops(crop1, crop2):
                return transform_after_crop(crop1), transform_after_crop(crop2)

            self.transforms = transforms.Compose([
                transform_crop,
                transforms.Lambda(lambda crops: apply_transforms_to_crops(*crops))
            ])
        else:
            self.transforms = transforms.Compose([
                transform_crop,
                transform_after_crop,
            ])
        
        test_transforms_list = [
            transforms.Resize(size=(crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if dataset == 'saycam':
            test_transforms_list.insert(1, transforms.RandomVerticalFlip(p=1.0))
        self.test_transforms = transforms.Compose(test_transforms_list)

        self.curr_base_fns = get_all_fns(self.dataset, subsample=self.subsample, start_epoch=0, class_length=self.class_length, label_path=self.label_path)
        if (not self.label_path) and self.class_length is not None:
            self.labels = (torch.arange(len(self.curr_base_fns)) // (self.class_length // self.subsample)).share_memory_()

    def __len__(self):
        return len(self.curr_base_fns)

    def __getitem__(self, idx):
        img_path, label, indices = self.curr_base_fns[idx]

        if (not self.label_path) and self.class_length is not None:
            label = self.labels[idx]

        try:
            img = Image.open(img_path)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(e)
            return self.__getitem__(idx+1)

        if self.ensure_overlap:
            img1, img2 = self.transforms(img)
        else:
            img1 = self.transforms(img)
            img2 = self.transforms(img)
        img = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)

        if self.return_indices:
            return img, label, indices
        else:
            return img, label

    def load_imgs_test(self, indices):
        all_imgs = []
        frame_ids = []
        for idx in indices:
            img_path, _, frame_id = self.curr_base_fns[idx]
            img = Image.open(img_path)    
            all_imgs.append(self.test_transforms(img))
            frame_ids.append(frame_id)
        all_imgs = torch.stack(all_imgs, dim=0)
        return all_imgs, frame_ids
    
    def set_labels(self, indices, labels):
        for idx, label in zip(indices, labels):
            self.labels[idx] = label

    def get_label(self, idx):
        if (not self.label_path) and self.class_length is not None:
            return self.labels[idx]
        else:
            return self.curr_base_fns[idx][1]

    def get_default_label(self, idx):
        return self.curr_base_fns[idx][1]

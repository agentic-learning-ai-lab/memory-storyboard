import os
import random
from PIL import ImageFilter
import torch
import math
import torchvision.transforms.functional as TF

class RandomResizedCropWithOverlap:
    def __init__(self, crop_size=(112, 112), scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        """
        Args:
            scale (tuple): range of the size of the origin size cropped.
            ratio (tuple): range of aspect ratio of the crops.
        """
        self.scale = scale
        self.ratio = ratio
        self.crop_size = crop_size

    def get_random_crop_params(self, img):
        """Generates random crop parameters based on the given scale and ratio."""
        width, height = img.size
        area = width * height

        for _ in range(10):
            # Get random scale and aspect ratio for the crop
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if crop_width <= width and crop_height <= height:
                top = random.randint(0, height - crop_height)
                left = random.randint(0, width - crop_width)
                return top, left, crop_height, crop_width

        # Fallback in case the above conditions fail
        crop_width = min(width, int(round(width * min(self.ratio))))
        crop_height = min(height, int(round(crop_width / min(self.ratio))))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        return top, left, crop_height, crop_width

    def crops_with_nonzero_overlap(self, img):
        """Generates two random crops that have non-zero overlap."""
        width, height = img.size

        # Get the first crop parameters
        top1, left1, crop_height1, crop_width1 = self.get_random_crop_params(img)

        # Ensure the second crop overlaps with the first crop
        # Randomize the position but ensure overlap
        while True:
            top2, left2, crop_height2, crop_width2 = self.get_random_crop_params(img)

            # Check for overlap (rectangles intersection)
            if not (top2 >= top1 + crop_height1 or top2 + crop_height2 <= top1 or
                    left2 >= left1 + crop_width1 or left2 + crop_width2 <= left1):
                break

        # Perform the crops
        crop1 = TF.resized_crop(img, top1, left1, crop_height1, crop_width1, self.crop_size)
        crop2 = TF.resized_crop(img, top2, left2, crop_height2, crop_width2, self.crop_size)

        return crop1, crop2

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            tuple: Two PIL Image objects representing the overlapping crops.
        """
        return self.crops_with_nonzero_overlap(img)

class GaussianBlur(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			sigma = random.random() * 1.9 + 0.1
			return img.filter(ImageFilter.GaussianBlur(sigma))
		else:
			return img

def count_seen_classes(video_dirs):
    seen_classes = [0]
    total = 0
    for video_dir in video_dirs:
        total += len(video_dir.split(','))
        seen_classes.append(total)
    return seen_classes

def get_all_fns(dataset, subsample=1, start_epoch=1, end_epoch=None, class_length=None, label_path=None):
    if dataset == 'saycam':
        root = '/path_to_saycam_frames'
    elif dataset == 'kcam':
        root = '/path_to_krishnacam_frames'
    else:
        raise ValueError
    
    epoch_meta = os.listdir(root)
    num_frames_meta = {}
    for video_dir_list in epoch_meta:
        video_dirs = video_dir_list.split(',')
        for video_dir in video_dirs:
            num_frames_meta[video_dir] = len(os.listdir(os.path.join(root, video_dir)))

    all_fns = []
    cum_frame_count = 0

    if end_epoch is None:
        end_epoch = len(epoch_meta)

    for epoch in range(start_epoch, end_epoch):
        
        if label_path:
            print("Processing epoch: ", epoch)
            labels = torch.load(os.path.join(label_path, f"labels_{epoch}.pt"))
            frames_count = 0
            print("maximum label: ", max(labels))

        video_dirs = epoch_meta[epoch].split(',')
        seen_classes = count_seen_classes(epoch_meta)
        for class_label, _dir in enumerate(video_dirs):
            num_imgs_class = num_frames_meta[_dir]

            if label_path:
                frames = [(os.path.join(root, _dir, '%06i.jpg' % (_frame)), labels[frames_count+_frame], frames_count+_frame) for _frame in range(0, num_imgs_class, subsample)]
                frames_count += num_imgs_class
            elif class_length is None:
                label = seen_classes[epoch] + class_label
                frames = [[os.path.join(root, _dir, '%06i.jpg' % (_frame)), label, cum_frame_count+_frame] for _frame in range(0, num_imgs_class, subsample)]
            else:
                frames = [[os.path.join(root, _dir, '%06i.jpg' % (_frame)), (cum_frame_count+_frame) // class_length, cum_frame_count+_frame] for _frame in range(0, num_imgs_class, subsample)]

            all_fns.extend(frames)
            cum_frame_count += num_imgs_class

        if label_path:
            assert frames_count == len(labels)

    return all_fns



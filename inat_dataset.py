import json
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

class INat2018(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(INat2018, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.split = split

        anno_filename = split + '2018.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']

    def __getitem__(self, index):
        path = os.path.join(self.root, self.images[index]['file_name'])
        target = self.annos[index]['category_id']

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)

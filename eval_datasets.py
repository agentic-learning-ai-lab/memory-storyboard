import torch
import os
from PIL import Image

class ImageList(torch.utils.data.Dataset):
    def __init__(self, root, list_file, pipeline):
        with open(list_file, 'r') as f:
            lines = f.readlines()        
      
        self.fns, self.labels = zip(*[l.strip().split() for l in lines])
        self.labels = [int(l) for l in self.labels]            
        self.fns = [os.path.join(root, fn) for fn in self.fns]  

        self.pipeline = pipeline
    
    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        path = self.fns[idx]              
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.pipeline(img)
        target = self.labels[idx]
        return img, target


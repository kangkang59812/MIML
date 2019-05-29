import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import json
from torchvision.datasets import CocoDetection


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, img_dir, anno_dir, coco_cat_id_to_class_ind_path):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.train_loader = CocoDetection(img_dir, anno_dir)
        with open(coco_cat_id_to_class_ind_path, 'r') as j:
            self.coco_cat_id_to_class_ind = json.load(j)
        self.transform = transforms.Compose([
            # transforms.RandomCrop(224, pad_if_needed=True),
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        image = self.train_loader[index][0]
        if self.transform is not None:
            image = self.transform(image)
        target = torch.zeros(80)
        # imgid = self.train_loader[index][1][0]['image_id']

        coco_cat_id = [str(item['category_id'])
                       for item in self.train_loader[index][1]]
        class_id = [self.coco_cat_id_to_class_ind[cat_id]
                    for cat_id in coco_cat_id]
        target[class_id] = 1.0
        target = torch.Tensor(target)
        return image, target

    def __len__(self):
        return len(self.train_loader)


def get_loader(img_dir, anno_dir, coco_cat_id_to_class_ind_path, batch_size=8, shuffle=True, num_workers=1):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(img_dir, anno_dir, coco_cat_id_to_class_ind_path)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader


if __name__ == "__main__":
    img_dir = '/home/lkk/datasets/coco2014/train2014'
    anno_dir = '/home/lkk/datasets/coco2014/annotations/instances_train2014.json'
    coco_cat_id_to_class_ind_path = '/home/lkk/code/MIML/coco_cat_id_to_class_ind.json'
    d = get_loader(img_dir, anno_dir, coco_cat_id_to_class_ind_path,
                   batch_size=8, shuffle=True, num_workers=0)

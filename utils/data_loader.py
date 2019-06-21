import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import json
import pdb
import cv2
from torch.utils.data import Dataset
import h5py
from torch import nn
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, origin_file, split, img_tags, vocab):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        if split in {'train', 'restval'}:
            self.split = ['train', 'restval']
        if split in {'val'}:
            self.split = ['val']
        if split in {'test'}:
            self.split = ['test']

        with open(origin_file, 'r') as j:
            self.origin_file = json.load(j)

        self.images_id = [(index, self.origin_file['images'][index]['imgid'])
                          for index in range(0, len(self.origin_file['images']))
                          if self.origin_file['images'][index]['split'] in self.split]

        with open(img_tags, 'r') as j:
            self.img_tags = json.load(j)

        with open(vocab, 'r') as j:
            self.vocab = json.load(j)
        self.transform = transforms.Compose([
            # transforms.RandomCrop(224, pad_if_needed=True),
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        word2id = self.vocab['word_map']
        real_index = self.images_id[index][0]  # index in origin file

        img_id = self.origin_file['images'][real_index]['imgid']
        path = self.origin_file['images'][real_index]['filepath'] + \
            '/'+self.origin_file['images'][real_index]['filename']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                print('ssssss')
                pdb.set_trace()
        # Convert caption (string) to word ids.
        tags = []
        random_choice = np.random.choice(self.img_tags[str(real_index)])
        t = list(map(str.lower, random_choice))
        tags = [word2id[token] for token in t]
        target = torch.zeros(len(word2id))
        target[list(map(lambda n:n-1, tags))]=1
        target = torch.Tensor(target)
        return image, target

    def __len__(self):
        return len(self.images_id)

    def image_at(self, index):
        real_index = self.images_id[index][0]  # index in origin file
        im_id = self.origin_file['images'][real_index]['imgid']
        path = self.origin_file['images'][real_index]['filepath'] + \
            '/'+self.origin_file['images'][real_index]['filename']
        im = Image.open(os.path.join(self.root, path)).convert('RGB')
        image_data = self.transform(im)
        tags = []
        random_choice = np.random.choice(self.img_tags[str(real_index)])
        t = list(map(str.lower, random_choice))
        tags = [self.vocab['word_map'][token] for token in t]
        # target = torch.zeros(len(self.vocab['word_map']))
        # target[list(map(lambda n:n-1, tags))]=1
        # target = torch.Tensor(target)
        # im = cv2.imread(os.path.join(self.root, path))
        return im, image_data, t


# Just for faster MIML
class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.train_hf = h5py.File(data_folder + '/train36.hdf5', 'r')
        self.train_features = self.train_hf['image_features']
        self.val_hf = h5py.File(data_folder + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']

        # Captions per image
        self.cpi = 5

        # Load encoded captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths
        # with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
        #     self.caplens = json.load(j)

        # Load bottom up image features distribution
        with open(os.path.join(data_folder, self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)

        with open(os.path.join(data_folder, 'WORDMAP_' + data_name + '.json'), 'r') as j:
            self.word_map = json.load(j)
        self.map_word = {v: k for k, v in self.word_map.items()}

        # for tags
        with open('/home/lkk/code/MIML/vocab.json', 'r') as j:
            self.vocab = json.load(j)
        self.words = list(self.vocab['word_map'].keys())
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):

        # The Nth caption corresponds to the (N // captions_per_image)th image
        objdet = self.objdet[i // self.cpi]

        # Load bottom up image features
        if objdet[0] == "v":
            img = torch.FloatTensor(self.val_features[objdet[1]])
        else:
            img = torch.FloatTensor(self.train_features[objdet[1]])
        img = img.unsqueeze(-1)
        caption = self.captions[i]
        # for index in caption:
        #     if index not in self.map_word.keys():
        tags = []
        caption = list(set(map(lambda x: self.map_word[x], caption)))
        remove = ['<start>', '<end>', '<pad>', '<unk>']
        for r in remove:
            if r in caption:
                caption.remove(r)

        for word in caption:
            if word in self.words:
                tags.append(self.vocab['word_map'][word])

        target = torch.zeros(len(self.vocab['word_map']))
        target[list(map(lambda n:n-1, tags))]=1
        target = torch.Tensor(target)

        return img, target

    def __len__(self):
        return self.dataset_size


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = [sum(cap).item() for cap in captions]
    return images, targets, lengths


def get_loader(root, origin_file, split, img_tags, vocab, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset

    coco = CocoDataset(root=root,
                       origin_file=origin_file,
                       split=split,
                       img_tags=img_tags,
                       vocab=vocab)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_loader2(data_folder, data_name, split, batch_size, shuffle, num_workers, transform=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset

    coco = CaptionDataset(data_folder, data_name, split, transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == "__main__":
    # data_folder = '/home/lkk/dataset'
    # data_name = 'coco_5_cap_per_img_5_min_word_freq'
    # d = get_loader2(data_folder=data_folder, data_name=data_name, split='TRAIN',
    #                 batch_size=8, shuffle=True, num_workers=0)
    # for i, (imgs, caps, caplens) in enumerate(d):

    #     # Move to GPU, if available
    #     imgs = imgs.to(device)
    #     caps = caps.to(device)
        
    #     print('')
    root = '/home/lkk/datasets/coco2014'
    origin_file = root+'/'+'dataset_coco.json'
    img_tags = './new_img_tags.json'
    voc = './vocab.json'
    d = get_loader(root=root, origin_file=origin_file, split='train',
                   img_tags=img_tags, vocab=voc, batch_size=1, shuffle=True, num_workers=0)
    c = CocoDataset(root=root,
                    origin_file=origin_file,
                    split='train',
                    img_tags=img_tags,
                    vocab=voc)
    im = c.image_at(0)
    for i, (imgs, tars, lens) in enumerate(d):
        images = imgs
        targets = tars
        lengths = lens
        print(images.shape, targets)
        if i == 2:
            break

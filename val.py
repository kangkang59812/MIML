import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from model.miml import MIML
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.nn.functional as F
import time
import random
import json
from utils.data_loader import get_loader
from tensorboardX import SummaryWriter
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def visualization(features):
    pass


def main(args):
    print("load vocabulary ...")
    # Load vocabulary wrapper
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    print("build data loader ...")
    # load data
    val_loader = get_loader(root=args.root, origin_file=args.caption_path, split=args.split,
                            img_tags=args.img_tags, vocab=args.vocab_path, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("build the models ...")
    # Build the models

    model = nn.DataParallel(MIML().cuda(), device_ids=[0, 1])
    model.load_state_dict(torch.load('./models/decoder-5-1771.ckpt'))

    model.eval()

    time_start = time.time()
    total_step = len(val_loader)

    with torch.no_grad():
        for epoch in range(args.num_epochs):
            for i, (imgs, tars, lens) in enumerate(val_loader):
                images = imgs.cuda()
                targets = tars.float().cuda()

                outputs = model(images)
                print('d')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/home/lkk/datasets/coco2014', help='root path')
    parser.add_argument('--model_path', type=str,
                        default='models2/', help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str,
                        default='./vocab.json', help='path for vocabulary wrapper')
    parser.add_argument('--split', type=str,
                        default='val', help='train/val/test')
    parser.add_argument('--img_tags', type=str,
                        default='./img_tags.json', help='imgages id and tags')
    parser.add_argument('--caption_path', type=str, default='/home/lkk/datasets/coco2014/dataset_coco.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=100,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1,
                        help='step size for saving trained models')

    # paraneters
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)

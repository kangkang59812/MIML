import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.nn.functional as F
import time
import random
import json
from utils.load_instance import get_loader
from tensorboardX import SummaryWriter
from model.miml import MIML
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def visualization(features):
    pass


def main(args):

    print("build data loader ...")
    # load data
    val_loader = get_loader(img_dir=args.img_dir, anno_dir=args.anno_dir, coco_cat_id_to_class_ind_path=args.cat_id2class_id,
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("build the models ...")
    # Build the models
    model = MIML(L=args.L, K=args.K, batch_size=args.batch_size)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(args.model_path))

    critiation = nn.BCELoss()
    critiation = nn.DataParallel(critiation, device_ids=[0, 1])
    time_start = time.time()
    total_step = len(val_loader)
    writer = SummaryWriter(log_dir='./loginstance')
    pre = torch.zeros(args.batch_size, args.L)
    with torch.no_grad():
        for i, (imgs, tars) in enumerate(val_loader):
            images = imgs.cuda()
            targets = tars.float().cuda()

            outputs = model(images)
            loss = critiation(outputs, targets).mean()
            pre = outputs >= args.threshold
            for j in range(args.batch_size):
                print('tar:',targets[j].nonzero())
                print('pre:', pre[j].nonzero())
            # Print log info
            if i % args.log_step == 0:
                time_end = time.time()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time:{}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), time_end-time_start))
                time_start = time_end
            writer.add_scalars(
                'val/loss', {'loss': loss.item()}, epoch*total_step+i)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='/home/lkk/datasets/coco2014/val2014', help='img path')
    parser.add_argument('--model_path', type=str,
                        default='/home/lkk/code/MIML/models/instance-1-647.ckpt', help='path for saving trained models')
    parser.add_argument('--anno_dir', type=str,
                        default='/home/lkk/datasets/coco2014/annotations/instances_val2014.json', help='path for anno')
    parser.add_argument('--cat_id2class_id', type=str,
                        default='/home/lkk/code/MIML/coco_cat_id_to_class_ind.json', help='path for id2id')
    parser.add_argument('--L', type=int, default=80)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    main(args)

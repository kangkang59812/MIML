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
from utils.data_loader import get_loader
from tensorboardX import SummaryWriter
from model.miml import MIML
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from utils.metric import compute_mAP
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualization(features):
    pass


def main(args):

    print("load vocabulary ...")
    # Load vocabulary wrapper
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    print("build data loader ...")
    # load data
    test_loader = get_loader(root=args.root, origin_file=args.caption_path, split=args.split,
                             img_tags=args.img_tags, vocab=args.vocab_path, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("build the models ...")
    # Build the models
    checkpoint = torch.load(args.model_path)

    model = MIML(L=args.L, K=args.K, batch_size=args.batch_size, base_model='resnet',
                 fine_tune=False)
    model.intermidate.load_state_dict(checkpoint['intermidate'])
    model.last.load_state_dict(checkpoint['last'])
    model.sub_concept_layer.load_state_dict(checkpoint['sub_concept_layer'])

    model = model.to(device)
    model.eval()
    critiation = nn.BCELoss()
    # critiation = nn.DataParallel(critiation, device_ids=[0, 1])
    time_start = time.time()
    total_step = len(test_loader)
    writer = SummaryWriter(log_dir='./log_test')

    hamming_loss_sum = 0.0
    mAp_sum = 0
    with torch.no_grad():
        for i, (imgs, tars) in enumerate(test_loader):
            images = imgs.cuda()
            targets = tars.float().cuda()
            pre = torch.zeros(args.batch_size, args.L)
            outputs = model(images)
            loss = critiation(outputs, targets)  # .mean()
            pre = outputs >= args.threshold

            # h_loss = hamming_loss(targets.cpu(), pre.cpu())
            mAp = compute_mAP(targets.cpu(), pre.cpu())

            # hamming_loss_sum += h_loss
            mAp_sum += mAp
            # for j in range(args.batch_size):
            #     print('tar:', targets[j].nonzero())
            #     print('pre:', pre[j].nonzero())
            # Print log info
            if i % args.log_step == 0:
                time_end = time.time()
                print('step :[{}/{}], mAp: [{:.4f}/{:.4f}], Time:{}'
                      .format(i, total_step, mAp, mAp_sum/(i+1), time_end-time_start))
                time_start = time_end
            writer.add_scalars(
                'metric', {'mAp': mAp}, i)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/home/lkk/datasets/coco2014', help='root path')
    parser.add_argument('--model_path', type=str,
                        default='/home/lkk/code/MIML/models/checkpoint_ResNet_epoch_22.pth.tar', help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str,
                        default='./vocab.json', help='path for vocabulary wrapper')
    parser.add_argument('--split', type=str,
                        default='test', help='train/val/test')
    parser.add_argument('--caption_path', type=str, default='/home/lkk/datasets/coco2014/dataset_coco.json',
                        help='path for train annotation json file')
    parser.add_argument('--img_tags', type=str,
                        default='./img_tags.json', help='imgages id and tags')
    parser.add_argument('--log_step', type=int, default=1,
                        help='step size for prining log info')

    parser.add_argument('--L', type=int, default=1024)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    main(args)

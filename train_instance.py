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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def visualization(features):
    pass


def main(args):

    print("build data loader ...")
    # load data
    train_loader = get_loader(img_dir=args.img_dir, anno_dir=args.anno_dir, coco_cat_id_to_class_ind_path=args.cat_id2class_id,
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("build the models ...")
    # Build the models
    model = MIML(L=args.L, K=args.K, batch_size=args.batch_size,
                 fine_tune=args.fine_tune)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])

    optimizer = torch.optim.Adam(
        [{'params': filter(lambda p: p.requires_grad, model.module.base_model.parameters()), 'lr': args.fine_tune_lr},
         {'params': model.module.sub_concept_layer.parameters(), 'lr': args.learning_rate}],
    )
    # add
    optimizer = nn.DataParallel(optimizer, device_ids=[0, 1])
    critiation = nn.BCELoss()
    critiation = nn.DataParallel(critiation, device_ids=[0, 1])
    time_start = time.time()
    total_step = len(train_loader)
    writer = SummaryWriter(log_dir='./loginstance')
    for epoch in range(args.num_epochs):
        for i, (imgs, tars) in enumerate(train_loader):
            images = imgs.cuda()
            targets = tars.float().cuda()

            outputs = model(images)
            loss = critiation(outputs, targets).mean()

            model.zero_grad()
            loss.backward()
            optimizer.module.step()

            # Print log info
            if i % args.log_step == 0:
                time_end = time.time()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time:{}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), time_end-time_start))
                time_start = time_end
            writer.add_scalars(
                'loss', {'loss': loss.item()}, epoch*total_step+i)
            # if i == 10:
            #     writer.close()
        # Save the model checkpoints
        if (epoch+1) % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(
                args.model_path, 'instance-{}-{}.ckpt'.format(epoch+1, i+1)))
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='/home/lkk/datasets/coco2014/train2014', help='img path')
    parser.add_argument('--model_path', type=str,
                        default='models/', help='path for saving trained models')
    parser.add_argument('--anno_dir', type=str,
                        default='/home/lkk/datasets/coco2014/annotations/instances_train2014.json', help='path for anno')
    parser.add_argument('--cat_id2class_id', type=str,
                        default='/home/lkk/code/MIML/coco_cat_id_to_class_ind.json', help='path for id2id')
    parser.add_argument('--split', type=str,
                        default='train', help='train/val/test')

    parser.add_argument('--log_step', type=int, default=1,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1,
                        help='step size for saving trained models')

    # paraneters
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--L', type=int, default=80)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--fine_tune', action="store_true", default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--fine_tune_lr', type=float, default=4e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)

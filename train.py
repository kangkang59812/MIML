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
from torch.optim import lr_scheduler
from utils.utils import save_checkpoint, adjust_learning_rate
from utils.metric import compute_mAP
from sklearn.metrics import f1_score, average_precision_score
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# def visualization(features):
#     pass


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print("load vocabulary ...")
    # Load vocabulary wrapper
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    print("build data loader ...")
    # load data
    train_loader = get_loader(root=args.root, origin_file=args.caption_path, split='train',
                              img_tags=args.img_tags, vocab=args.vocab_path, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = get_loader(root=args.root, origin_file=args.caption_path, split='val',
                            img_tags=args.img_tags, vocab=args.vocab_path, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("build the models ...")
    # Build the models
    model = MIML(L=args.L, K=args.K, batch_size=args.batch_size,
                 fine_tune=args.fine_tune)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = torch.optim.Adam(
        [{'params': filter(lambda p: p.requires_grad, model.module.base_model.parameters()), 'lr': args.fine_tune_lr},
         {'params': model.module.sub_concept_layer.parameters(), 'lr': args.learning_rate}], weight_decay=0.0005
    )

    optimizer_SGD = torch.optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, model.module.base_model.parameters()), 'lr': args.fine_tune_lr},
         {'params': model.module.sub_concept_layer.parameters(), 'lr': args.learning_rate}],
    )

    optimizer_Momentum = torch.optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, model.module.base_model.parameters()), 'lr': args.fine_tune_lr},
         {'params': model.module.sub_concept_layer.parameters(), 'lr': args.learning_rate}],
        momentum=0.8
    )

    optimizer_RMSprop = torch.optim.RMSprop(
        [{'params': filter(lambda p: p.requires_grad, model.module.base_model.parameters()), 'lr': args.fine_tune_lr},
         {'params': model.module.sub_concept_layer.parameters(), 'lr': args.learning_rate}],
        alpha=0.9
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    # optimizer = nn.DataParallel(optimizer, device_ids=[0, 1])
    criterion = nn.BCELoss()
    # criterion = nn.DataParallel(criterion, device_ids=[0, 1])
    best_ac = 0
    epochs_since_improvement = 0
    writer = SummaryWriter(log_dir='./log2')

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        # at first I didn't add weight_decay
        optimizer.__setattr__('weight_decay', 0.0005)
        best_ac = checkpoint['accuracy']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        ep = checkpoint['epoch']
    for epoch in range(args.num_epochs):

        if epoch < ep:
            continue
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        # if epochs_since_improvement == 20:
        #     break

        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            # scheduler.step()
            lr1 = optimizer.param_groups[0]['lr']
            lr2 = optimizer.param_groups[1]['lr']
            optimizer = torch.optim.Adam(
                [{'params': filter(lambda p: p.requires_grad, model.module.base_model.parameters()), 'lr': lr1},
                 {'params': model.module.sub_concept_layer.parameters(), 'lr': lr2}], weight_decay=0.0005
            )
            adjust_learning_rate(optimizer, 0.8)

        # train(args, train_loader=train_loader, model=model, criterion=criterion,
        #       optimizer=optimizer, epoch=epoch, writer=writer)

        accuracy = validate(args, val_loader=val_loader, model=model, criterion=criterion,
                            epoch=epoch, writer=writer)

        is_best = accuracy > best_ac
        best_ac = max(accuracy, best_ac)
        if not is_best:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        # save model
        save_checkpoint(data_name='model', epoch=epoch, epochs_since_improvement=epochs_since_improvement, model=model,
                        optimizer=optimizer, accuracy=accuracy, is_best=is_best)
    writer.close()


def train(args, train_loader, model, criterion, optimizer, epoch, writer):
    model.train()
    total_step = len(train_loader)
    mAp_sum = 0
    f1_sum = 0
    time_start = time.time()
    save_targets = None
    save_outputs = None
    mAp_by_label = 0
    f1_by_label = 0
    sum_loss = 0
    for i, (imgs, tars, lens) in enumerate(train_loader):
        images = imgs.cuda()
        targets = tars.float().cuda()

        outputs = model(images)
        loss = criterion(outputs, targets)  # .mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 0:
            save_targets = torch.cat([targets.detach().cpu()])
            save_outputs = torch.cat([outputs.detach().cpu()])
        else:
            save_targets = torch.cat([save_targets, targets.detach().cpu()])
            save_outputs = torch.cat([save_outputs, outputs.detach().cpu()])
        # mAp f1 , by samples
        mAp = average_precision_score(
            targets.detach().cpu(), outputs.detach().cpu(), average='samples')
        f1 = f1_score(targets.detach().cpu(),
                      outputs.detach().cpu() >= args.threthold, average='samples')
        mAp_sum += mAp
        f1_sum += f1
        sum_loss += loss.item()
        # optimizer.module.step()
        # Print log info
        if i % args.log_step == 0:
            time_end = time.time()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, mAp:[{:.4f}/{:.4f}], F1:[{:.4f}/{:.4f}],Time:{}'
                  .format(epoch, args.num_epochs, i, total_step, loss.item(), mAp, mAp_sum/(i+1), f1, f1_sum/(i+1), time_end-time_start))
            time_start = time_end
        writer.add_scalars(
            'train:  ', {'loss': loss.item(), 'mAp': mAp, 'F1': f1}, epoch*total_step+i)

    mAp_by_label = average_precision_score(save_targets, save_outputs)
    f1_by_label = f1_score(save_targets, save_outputs >=
                           args.threthold, average='macro')
    writer.add_scalars(
        'train_metric:  ', {'loss': sum_loss/total_step, 'mAp_label': mAp_by_label, 'F1': f1_by_label}, epoch)


def validate(args, val_loader, model, criterion, epoch, writer):
    model.eval()
    total_step = len(val_loader)
    time_start = time.time()
    mAp_by_label = 0
    f1_by_label = 0
    sum_loss = 0
    save_targets = None
    save_outputs = None
    with torch.no_grad():
        for i, (imgs, tars, lens) in enumerate(val_loader):
            images = imgs.cuda()
            targets = tars.float().cuda()
            pre = torch.zeros(args.batch_size, args.L)
            outputs = model(images)
            loss = criterion(outputs, targets)  # .mean()

            if i == 0:
                save_targets = torch.cat([targets.detach().cpu()])
                save_outputs = torch.cat([outputs.detach().cpu()])
            else:
                save_targets = torch.cat(
                    [save_targets, targets.detach().cpu()])
                save_outputs = torch.cat(
                    [save_outputs, outputs.detach().cpu()])

            sum_loss += loss.item()
            if i % args.log_step == 0:
                time_end = time.time()
                print('step :[{}/{}], loss:{}, Time:{}'
                      .format(i, total_step, loss, time_end-time_start))
                time_start = time_end
        mAp = average_precision_score(save_targets, save_outputs, average='samples')
        f1 = f1_score(save_targets, save_outputs >= args.threthold, average='samples')

        mAp_by_label = average_precision_score(save_targets, save_outputs)
        f1_by_label = f1_score(save_targets, save_outputs >= args.threthold, average='macro')

        writer.add_scalars(
            'val_metric_by_sample', {'loss': sum_loss/total_step, 'mAp': mAp, 'f1': f1}, epoch)

        writer.add_scalars(
            'val_metric_by_label', {'loss': sum_loss/total_step, 'mAp': mAp_by_label, 'f1': f1_by_label}, epoch)

    return mAp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/home/lkk/datasets/coco2014', help='root path')
    parser.add_argument('--model_path', type=str,
                        default='models/', help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str,
                        default='./vocab.json', help='path for vocabulary wrapper')
    parser.add_argument('--img_tags', type=str,
                        default='./img_tags.json', help='imgages id and tags')
    parser.add_argument('--caption_path', type=str, default='/home/lkk/datasets/coco2014/dataset_coco.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=1,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1,
                        help='step size for saving trained models')
    parser.add_argument('--checkpoint', type=str, default='/home/lkk/code/MIML/models/BEST_checkpoint_model_epoch_5.pth.tar',
                        help='load checkpoint')

    # paraneters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--L', type=int, default=1024)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--fine_tune', action="store_true", default=True)
    parser.add_argument('--num_workers', type=int,
                        default=0)  # 0 only for debugging
    parser.add_argument('--fine_tune_lr', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--threthold', type=float, default=0.5)
    args = parser.parse_args()
    main(args)

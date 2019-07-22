
from utils.utils import plot_instance_attention, plot_instance_probs_heatmap
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils.data_loader import CocoDataset
from model.miml import MIML
import json
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
features = torch.empty(1, 64, 1024).to(device)
instance_probs = None


def hook(module, input, ouput):
    global features, instance_probs
    features = torch.empty(1, 64, 1024).to(device)
    features.copy_(ouput.data)
    features = features.permute(0, 2, 1).reshape(-1, 1024, 1, 64)
    instance_probs = features.permute(0, 3, 1, 2)[:, :, :, 0].squeeze().cpu()
    # print("instance_probs.shape=", instance_probs.shape)

    # plot instance label score
    plot_instance_probs_heatmap(instance_probs, './1.jpg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str,
                        default="/home/lkk/code/MIML/models/checkpoint_ResNet_epoch_22.pth.tar")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    
    root = '/home/lkk/datasets/coco2014'
    origin_file = root+'/'+'dataset_coco.json'
    img_tags = './img_tags.json'
    voc = './vocab.json'
    dataset = CocoDataset(root=root,
                          origin_file=origin_file,
                          split='test',
                          img_tags=img_tags,
                          vocab=voc)
    choose = np.random.randint(0, len(dataset), 10)
    with open(voc, 'r') as j:
        vocab = json.load(j)

    cls_names = vocab['map_word']

    checkpoint = torch.load(args.model)
    model = MIML(L=1024, K=20, batch_size=8, base_model='resnet',
                 fine_tune=False)
    model.intermidate.load_state_dict(checkpoint['intermidate'])
    model.last.load_state_dict(checkpoint['last'])
    model.sub_concept_layer.load_state_dict(checkpoint['sub_concept_layer'])

    model = model.to(device)
    model.eval()
    for it in choose:
        im, image_data, target = dataset.image_at(it)

        # heat map
        handle = model.sub_concept_layer.softmax1.register_forward_hook(
            hook)
        label_id_list = np.where(model(image_data.unsqueeze(
            0).cuda()).cpu().detach().numpy() > 0.5)[0]
        handle.remove()
        label_name_list = [cls_names[str(i+1)] for i in label_id_list]
        instance_points, instance_labels = [], []
        for _i, label_id in enumerate(label_id_list):
            max_instance_id = np.argmax(instance_probs[:, label_id])
            conv_y, conv_x = max_instance_id / 8, max_instance_id % 8
            instance_points.append(((conv_x * 32 + 4), (conv_y * 32 + 4)))
            instance_labels.append(label_name_list[_i])
        im_plot = cv2.resize(np.array(im), (256, 256)).astype(
            np.uint8)[:, :, (0, 1, 2)]
        plot_instance_attention(im_plot, instance_points,
                                instance_labels, save_path='./vis_5/'+str(it)+'.jpg')
        print(target)
        print(instance_labels)
        print('****************')

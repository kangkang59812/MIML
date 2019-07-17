import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from matplotlib import cm
import torch
import os


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is {} and {}".format(
        optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, args, epoch, epochs_since_improvement, model, optimizer, accuracy, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    if args.mGPUs:
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'accuracy': accuracy,
                 'model': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
    else:
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'accuracy': accuracy,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
    filename = os.path.join('/home/lkk/code/MIML/models',
                            'checkpoint_' + data_name + '_epoch_'+str(epoch)+'.pth.tar')

    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        filename = os.path.join('/home/lkk/code/MIML/models',
                                'BEST_checkpoint_' + data_name + '_epoch_'+str(epoch)+'.pth.tar')
        torch.save(state, filename)


def plot_instance_attention(im, instance_points, instance_labels, save_path=None):
    """
    Arguments:
        im (ndarray): shape = (3, im_width, im_height)
            the image array
        instance_points: List of (x, y) pairs
            the instance's center points
        instance_labels: List of str
            the label name of each instance
    """
    fig, (ax, ax2) = plt.subplots(1, 2)
    ax.imshow(im)
    ax2.imshow(im)
    for i, (x_center, y_center) in enumerate(instance_points):
        label = instance_labels[i]
        center = plt.Circle((x_center, y_center), 10, color="r", alpha=0.5)
        ax.add_artist(center)
        ax.text(x_center, y_center, str(label), fontsize=18,
                bbox=dict(facecolor="blue", alpha=0.7), color="white")
    if save_path:
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))
        fig.savefig(save_path)
    else:
        plt.show()


def plot_instance_probs_heatmap(instance_probs, save_path=None):
    """
    Arguments:
        instance_probs (ndarray): shape = (n_instances, n_labels)
            the probability distribution of each instance
    """
    n_instances, n_labels = instance_probs.shape
    fig, ax = plt.subplots()
    ax.set_title("Instance-Label Scoring Layer Visualized")

    cax = ax.imshow(instance_probs, vmin=0, vmax=1, cmap=cm.hot,
                    aspect=float(n_labels) / n_instances)
    cbar_ticks = list(np.linspace(0, 1, 11))
    cbar = fig.colorbar(cax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(map(str, cbar_ticks))

    if save_path:
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))
        fig.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    instance_probs = np.random.random((196, 80)) * 0.5
    plot_instance_probs_heatmap(instance_probs, './test/1.jpg')

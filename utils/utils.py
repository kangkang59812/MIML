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


def save_checkpoint(data_name, epoch, epochs_since_improvement, model, optimizer, accuracy, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'accuracy': accuracy,
             'model': model,
             'optimizer': optimizer}
    filename = os.path.join('/home/lkk/code/MIML/models',
                            'checkpoint_' + data_name + '_epoch_'+str(epoch)+'.pth.tar')

    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        filename = os.path.join('/home/lkk/code/MIML/models',
                                'BEST_checkpoint_' + data_name + '_epoch_'+str(epoch)+'.pth.tar')
        torch.save(state, filename)

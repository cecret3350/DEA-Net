from matplotlib import pyplot as plt
import numpy as np
import os


def plot_loss_log(loss_log, epoch, loss_dir):
    axis = np.linspace(1, epoch, epoch)
    for key in loss_log.keys():
        label = '{} Loss'.format(key)
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, np.array(loss_log[key]))
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(loss_dir, 'loss_{}.pdf'.format(key)))
        plt.close(fig)


def plot_psnr_log(psnr_log, epoch, psnr_dir):
    axis = np.linspace(1, epoch, epoch)
    label = 'PSNR'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(psnr_log))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(psnr_dir, 'psnr.pdf'))
    plt.close(fig)
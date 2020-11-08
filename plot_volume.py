import matplotlib.pyplot as plt
import numpy as np
import imageio


def save_slice(arr, slice_num, fname):
    plt.imsave(fname, arr[0, slice_num, :, :, 0])
    return


def plot_gif(data_arr, mask_arr, fname):
    mask_arr = mask_arr > 0.05
    combined_arr = np.concatenate((data_arr, mask_arr), axis=-1)
    imageio.mimsave(fname, combined_arr, duration=0.0001)
    return



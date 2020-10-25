import matplotlib.pyplot as plt
import numpy as np
import imageio
import Reza_functions as rf

# Courtesy of Shashank

def plot_gif(data_arr, mask_arr, fname):
    mask_arr = rf.hu_to_grayscale(mask_arr)
    combined_arr = np.concatenate((data_arr, mask_arr), axis=-1)
    imageio.mimsave(fname, combined_arr, duration=0.0001)
    return

if __name__ == "__main__":
    plot_gif(np.load("processed_data/my_data_test.npy"), np.load("output/my_results.npy"), "my_sample_results.gif")
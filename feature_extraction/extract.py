from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np


def extract_lungs(scan_data, lung_mask):
    """ Extracts only lung data from scans using initial scan and segmentation mask """
    # Setting empty space intensity for scan to zero - this assumes that the first value in the first array, index (0,
    # 0, 0), is the empty space intensity for the whole scan
    empty_space_val = scan_data[0, 0]
    scan_data[scan_data == empty_space_val] = 0.0

    # Removing low intensity values < 25
    scan_data[scan_data < 25] = 0.0

    # Making edges have value 1
    lung_mask[:, :, :25] = 1.0
    lung_mask[:, :, -25:] = 1.0
    
    # The mask values are 0 at areas of interest - we want them to be 1
    lung_mask = 1.0 - lung_mask
    lung_mask = (lung_mask * 255).astype(int)

    # Mutliplying scan data and lung segmentation map
    lung_data = scan_data * lung_mask
    return lung_data


def lung_volume(lung_mask):
    """ Extracts lung volume from scans using segmentation mask """
    return np.sum(lung_mask)


if __name__ == "__main__":
    segmentation_fname = "tmp/ID00378637202298597306391.npy"
    #  data_fname = "segmentation/processed_data/my_data_test.npy"

    mask = np.load(segmentation_fname)
    #  data = np.load(data_fname)

    lung_data = lung_volume(mask)
    plt.imsave("test.png", mask[15, :, :])
    print(lung_data)

    #  plt.imsave("test.png", np.concatenate((lung_data[15, :, :], (1.0 - mask[15, :, :])*255.0), axis=1))


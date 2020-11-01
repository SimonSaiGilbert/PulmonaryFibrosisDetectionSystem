from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os


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


def load_dicom_data(fname):
    data = pydicom.dcmread(fname)
    return data

def lung_volume(lung_mask, dicom_data):
    """ Extracts lung volume from scans using segmentation mask (unit is mm^3) """
    slice_thickness = float(dicom_data[0x0018, 0x0050].value)
    pixel_spacing = [float(x) for x in dicom_data[0x0028, 0x0030].value]
    voxel_size = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    return np.sum(lung_mask)*voxel_size


if __name__ == "__main__":
    segmentation_dir = "/projectnb/ece601/F-PuPS/kaggle/output_pipeline/train/" 
    dicom_dir = "/projectnb/ece601/F-PuPS/kaggle/data/train"
    patient_id = "ID00007637202177411956430"

    segmentation_fname = os.path.join(segmentation_dir, "{}.npy".format(patient_id))
    dicom_dir = os.path.join(dicom_dir, patient_id)
    dicom_files = sorted(os.listdir(dicom_dir), key=lambda x: int(x.replace(".dcm", "")))
    dicom_files = [os.path.join(dicom_dir, x) for x in dicom_files]

    dicom_data = [load_dicom_data(fname) for fname in dicom_files]
    mask = np.load(segmentation_fname)

    for idx, d in enumerate(dicom_data):
        lung_data = lung_volume(mask[idx, :, :], d)

    #  plt.imsave("test.png", mask[15, :, :])
    #  print(lung_data)

    #  plt.imsave("test.png", np.concatenate((lung_data[15, :, :], (1.0 - mask[15, :, :])*255.0), axis=1))


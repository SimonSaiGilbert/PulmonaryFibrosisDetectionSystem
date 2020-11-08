import pandas as pd
import numpy as np
import pydicom
import glob
import os
import re


def crop_center(data, cropx, cropy):
    '''Crop all images in the data'''
    z,y,x = data.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    
    return data[:, starty:starty+cropy, startx:startx+cropx]

def crop_and_normalize_dicom(img, hu=[-1200., 600.]):
    lungwin = np.array(hu)
    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    #  newimg = (newimg * 255).astype('uint8')
    newimg = newimg.astype(np.float)
    newimg = crop_center(newimg,512,512)
    return newimg


def load_scan(data_dir):
    """ Function to load a single scan represented by multiple DICOM files in a directory """

    # Finding all DICOM files in specified directory
    dcm_files = glob.glob(os.path.join(data_dir, "*.dcm"))
    
    # Sorting the files in ascending order appropriately
    dcm_files.sort(key=lambda f: int(re.sub("\D", "", f)))

    # Loading dcm files
    data_list = []
    for f in dcm_files:
        data_list.append(pydicom.dcmread(f))

    # Concatenating all the arrays together to form a [Num Images, rows, cols] size array
    data = np.stack([x.pixel_array for x in data_list])

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    data[data <= -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(data.shape[0]):
        
        intercept = data_list[slice_number].RescaleIntercept
        slope = data_list[slice_number].RescaleSlope
        
        if slope != 1:
            data[slice_number] = slope * data[slice_number].astype(np.float64)
            data[slice_number] = data[slice_number].astype(np.int16)
            
        data[slice_number] += np.uint16(intercept)
    data = crop_and_normalize_dicom(data)
    return data


def load_train_dataset(data_dir, label_file):
    """ Function to load train dataset """

    # Getting all scan names
    scans = os.listdir(data_dir)

    # Getting labels
    labels = pd.read_csv(label_file)
    print(labels)
    return


if __name__ == "__main__":
    data_dir = "/projectnb/ece601/F-PuPS/kaggle/data/train"
    label_file = "/projectnb/ece601/F-PuPS/kaggle/data/train.csv"

    data, labels = load_train_dataset(data_dir, label_file)



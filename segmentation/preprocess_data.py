import numpy as np
import pydicom
import glob
import os
import re
import Reza_functions as rf

# Courtesy of Shashank

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

def load_data(data_dir):
    """ Function to load all DICOM files in directory """

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
            
        data[slice_number] += np.int16(intercept)
 
    data = crop_and_normalize_dicom(data)
    
    for slice_number in range(data.shape[0]):
        data[slice_number] = rf.hu_to_grayscale(data[slice_number])
    
    return data

def save_data(data):
    '''Save data to npy file'''

    folder = './processed_data/'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    np.save(folder+'my_data_test', data)

if __name__ == "__main__":
 #   data_dir = "ID00419637202311204720264/"
    data_dir = "/projectnb/ece601/F-PuPS/kaggle/data/test/ID00419637202311204720264/"
    save_data(load_data(data_dir))    
else:
    data_dir = "/projectnb/ece601/F-PuPS/kaggle/data/test/ID00419637202311204720264/"
    save_data(load_data(data_dir))

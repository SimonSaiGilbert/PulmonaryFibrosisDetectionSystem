from plot_volume import plot_gif
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pydicom
import glob
import os
import re


def unet3d():
    in_layer = tf.keras.layers.Input((None, None, None, 1))
    bn = tf.keras.layers.BatchNormalization()(in_layer)
    cn1 = tf.keras.layers.Conv3D(8, 
                kernel_size = (1, 5, 5), 
                padding = 'same',
                activation = 'relu')(bn)
    cn2 = tf.keras.layers.Conv3D(8, 
                kernel_size = (3, 3, 3),
                padding = 'same',
                activation = 'linear')(cn1)
    bn2 = tf.keras.layers.BatchNormalization()(cn2)      
    bn2 = tf.keras.layers.Activation('relu')(bn2)

    dn1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(bn2)
    cn3 = tf.keras.layers.Conv3D(16, 
                kernel_size = (3, 3, 3),
                padding = 'same',
                activation = 'linear')(dn1)
    bn3 = tf.keras.layers.BatchNormalization()(cn3)
    bn3 = tf.keras.layers.Activation('relu')(bn3)

    dn2 = tf.keras.layers.MaxPooling3D((1, 2, 2))(bn3)
    cn4 = tf.keras.layers.Conv3D(32, 
                kernel_size = (3, 3, 3),
                padding = 'same',
                activation = 'linear')(dn2)
    bn4 = tf.keras.layers.BatchNormalization()(cn4)
    bn4 = tf.keras.layers.Activation('relu')(bn4)

    up1 = tf.keras.layers.Conv3DTranspose(16, 
                        kernel_size = (3, 3, 3),
                        strides = (1, 2, 2),
                        padding = 'same')(bn4)

    cat1 = tf.keras.layers.concatenate([up1, bn3])

    up2 = tf.keras.layers.Conv3DTranspose(8, 
                        kernel_size = (3, 3, 3),
                        strides = (2, 2, 2),
                        padding = 'same')(cat1)

    pre_out = tf.keras.layers.concatenate([up2, bn2])

    pre_out = tf.keras.layers.Conv3D(1, 
                kernel_size = (1, 1, 1), 
                padding = 'same',
                activation = 'sigmoid')(pre_out)

    pre_out = tf.keras.layers.Cropping3D((1, 2, 2))(pre_out) # avoid skewing boundaries
    out = tf.keras.layers.ZeroPadding3D((1, 2, 2))(pre_out)
    
    model = tf.keras.models.Model(inputs = [in_layer], outputs = [out])
    model.summary()
    
    return model


def load_model(weights_path):
    model = unet3d()
    model.load_weights(weights_path)
    return model


def run_inference(data, weights_path):
    model = load_model(weights_path)
    pred = model.predict(data)
    return pred


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
    return data


if __name__ == "__main__":
    data_dir = "/projectnb/ece601/F-PuPS/kaggle/data/test/ID00419637202311204720264/"
    #  data_dir = "/projectnb/ece601/F-PuPS/kaggle/data/train/ID00007637202177411956430"
    data = load_data(data_dir)[None, :, :, :, None]
    output_data = run_inference(data, "convlstm_model_best_weights.hdf5")

    data = data[0, :, :, :, 0]
    output_data = output_data[0, :, :, :, 0]
    plot_gif(data, output_data, "example_outputs.gif")

    #  plt.imsave("tmp_input.png", data[0, 14, :, :, 0])
    #  plt.imsave("tmp_output.png", output_data[0, 14, :, :, 0] > 0.05)


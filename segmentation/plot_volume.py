import matplotlib.pyplot as plt
import numpy as np
import imageio
import Reza_functions as rf
import glob
import os
from datetime import date

# Courtesy of Shashank

def plot_gif(data_arr, mask_arr, fname):
    mask_arr = rf.hu_to_grayscale(mask_arr)
    combined_arr = np.concatenate((data_arr, mask_arr), axis=-1)
    imageio.mimsave(fname, combined_arr, duration=0.0001)
    return

def plot_all_patients():
    raw_data_path = "/projectnb/ece601/F-PuPS/kaggle/data/test/"
    data_path = "/projectnb/ece601/F-PuPS/kaggle/preprocessed_data/test/"
    mask_path = "/projectnb/ece601/F-PuPS/kaggle/output_pipeline/test/"
    output_path = "test_patient_gifs/" + str(date.today()) + "/"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    patients = glob.glob(raw_data_path + "ID*")    
    for patient in patients:
        input_path, patient_id = patient.rsplit('/',1)
        plot_gif(np.load(data_path + patient_id + ".npy"), np.load(mask_path + patient_id + ".npy"), output_path + patient_id + ".gif")

if __name__ == "__main__":
    plot_gif(np.load("processed_data/my_data_test.npy"), np.load("output/my_results.npy"), "my_sample_results.gif")
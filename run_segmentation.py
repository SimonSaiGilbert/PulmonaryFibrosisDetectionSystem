from loader.loader import load_scan
from segmentation import models
from fire import Fire
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def main(data_dir, save_dir, weights_dir):
    """ Function to run segmentation network on all patients scans in folder """
    for idx, patient_id in enumerate(tqdm(os.listdir(data_dir))):
        scan_data = load_scan(os.path.join(data_dir, patient_id))

        scan_data = np.expand_dims(scan_data, axis=-1)
        model = models.BCDU_net_D3(input_size=(512,512,1))
        model.summary()
        model.load_weights(weights_dir)
        predictions = model.predict(scan_data, batch_size=16, verbose=1)

        predictions = np.squeeze(predictions)
        predictions = np.where(predictions > 0.5, 1, 0)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, patient_id), predictions)

        if idx >= 0:
            break


if __name__ == "__main__":
    Fire(main)

####################################  Load Data #####################################
#  folder    = "projectnb/ece601/F-PuPS/kaggle/data/train/"
#
#  for fname in os.listdir(folder):
#      patient_id = fname
#      dicom_folder = os.path.join(folder, fname)
#      scan_data =
#  te_data   = np.load(folder+'my_data_test.npy')
#
#  te_data  = np.expand_dims(te_data, axis=3)
#
#  print('Dataset loaded')

####################################  Run Model #####################################
#  te_data2 = te_data /255.
#  model = M.BCDU_net_D3(input_size = (512,512,1))
#  model.summary()
#  model.load_weights('weight_lung')
#  predictions = model.predict(te_data2, batch_size=2, verbose=1)
#
#  predictions = np.squeeze(predictions)
#  predictions = np.where(predictions>0.5, 1, 0)

####################################  Save Data #####################################
#  output_folder = 'output/'
#  if not os.path.exists(output_folder):
#      os.makedirs(output_folder)
#
#  np.save(output_folder+'my_results', predictions)

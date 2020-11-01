from segmentation.preprocess_data import load_data
from segmentation import models
from fire import Fire
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def main(data_dir, save_dir, weights_dir):
    """ Function to run segmentation network on all patients scans in folder """
    model = models.BCDU_net_D3(input_size=(512,512,1))
    model.summary()
    model.load_weights(weights_dir)

    for idx, patient_id in enumerate(tqdm(os.listdir(data_dir))):
        scan_data = load_data(os.path.join(data_dir, patient_id))
        scan_data = scan_data / 255.
        scan_data = np.expand_dims(scan_data, axis=-1)
        predictions = model.predict(scan_data, batch_size=16, verbose=1)

        predictions = np.squeeze(predictions)
        predictions = np.where(predictions > 0.5, 1, 0)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, patient_id), predictions)

        plt.imsave("test.png", predictions[15, :, :])

        if idx >= 0:
            break


if __name__ == "__main__":
    Fire(main)


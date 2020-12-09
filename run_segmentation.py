from segmentation.preprocess_data import load_data
from segmentation import models
from fire import Fire
from tqdm import tqdm
from fvc_model.neural_network import segmentation_model, transfer_weights
from segmentation.gui_segment_pipeline import segmentation_fn

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def main(data_dir, save_dir, weights_dir):
    """ Function to run segmentation network on all patients scans in folder """
    # model = models.BCDU_net_D3(input_size=(128,64,64,1))
    # model.summary()
    # print('weights_dir: ', weights_dir)
    # model.load_weights(weights_dir)
    model = segmentation_model((30,512,512,1), backbone_weights=weights_dir)
    model.summary()
    segmentation_fn(data_dir)


    # for idx, patient_id in enumerate(tqdm(os.listdir(data_dir))):
    #     try:
    #         print("file path: ", os.path.join(data_dir, patient_id))
    #         scan_data = load_data(os.path.join(data_dir, patient_id))
    #         scan_data = scan_data / 255.
    #         scan_data = np.expand_dims(scan_data, axis=-1)
    #         print("data shape: ", scan_data.shape)
    #         predictions = model.predict(scan_data, batch_size=1, verbose=1)
    #         predictions = np.squeeze(predictions)
    #         predictions = np.where(predictions > 0.5, 1, 0)

    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         np.save(os.path.join(save_dir, patient_id), predictions)

    #         plt.imsave("test.png", predictions[15, :, :])

    #         if idx >= 0:
    #             break
    #     except:
    #         print("ayyyy lmao")


if __name__ == "__main__":
    Fire(main)


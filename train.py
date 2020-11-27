from csv_reader.csv_to_dict import csv_to_dict
from tensorflow.keras.optimizers import Adam
from fvc_model.neural_network import nn_model
from loader.loader import load_scan
from tqdm import tqdm

import tensorflow as tf
import scipy.signal
import numpy as np
import os


sex_to_int = {
    "Male": 0,
    "Female": 1,
}


smoking_status_to_int = {
    "Never": 0,
    "Ex-smoker": 1,
    "Currently": 2,
}


class ModelTrainer(object):
    def __init__(self, model: tf.keras.models.Model, run_name: str):
        self.model = model
        self.run_name = run_name

    def train(self, lr: float, train_data: dict, test_data: dict, epochs: int=100):
        self.model.compile(optimizer=Adam(lr=lr), loss="MSE")    
        history = self.model.fit(
            train_data["data"], train_data["label"], 
            batch_size=1, 
            epochs=epochs,
            #  validation_data=(np.concatenate(test_data["data"][0], axis=0), np.zeros((len(test_data["data"][0]), 512, 512, 1)))
        )
        #  history = self.model.fit(
        #      train_data["data"], train_data["label"],
        #      batch_size=1,
        #      epochs=epochs,
        #      validation_data=(test_data["data"], test_data["label"])
        #  )
        return

    def eval(self, test_arr: np.ndarray):
        predictions = model.predict(test_arr)
        return predictions


def pad_arr(input_arr, target_dim_size, dim):
    """ Zero-pads array to size specified by target_dim_size  """
    zero_arr_size = []

    for idx, dim_size in enumerate(input_arr.shape):
        if dim != idx:
            zero_arr_size.append(dim_size)
        else:
            zero_arr_size.append(target_dim_size - dim_size)
    scan_data = np.concatenate((input_arr, np.zeros((zero_arr_size))), axis=dim)
    return scan_data


def downsample_arr(input_arr, target_dim_size, dim):
    output_arr = scipy.signal.resample_poly(input_arr, down=input_arr.shape[dim], up=target_dim_size, axis=dim)
    return output_arr


def correct_dim_size(arr, target_dim_size, dim):
    """ Pads or downsamples array to size specified by target_dim_size in dimension specified by dim """
    if arr.shape[dim] > target_dim_size:
        arr = downsample_arr(arr, target_dim_size, dim)
    elif arr.shape[dim] < target_dim_size:
        arr = pad_arr(arr, target_dim_size, dim)
    return arr


def load_dataset(scan_data_dir, csv_dir, target_scan_size):
    dataset = {"data": [[], []], "label": []}
    csv_data = csv_to_dict(csv_dir, pad_with_zeros=True)

    if "train" in scan_data_dir:
        csv_data = csv_data[0]
    elif "test" in scan_data_dir:
        csv_data = csv_data[1]

    for pt_idx, patient_id in enumerate(tqdm(os.listdir(scan_data_dir))):
        scan_data = load_scan(os.path.join(scan_data_dir, patient_id))
        slice_dim = scan_data.shape[1]

        # Cropping or padding to appropriate size
        for axis_idx, _ in enumerate(scan_data.shape):
            scan_data = correct_dim_size(scan_data, target_scan_size[axis_idx], axis_idx)

        scan_data = scan_data / 255.
        scan_data = np.expand_dims(scan_data, axis=-1)

        pt_data = csv_data[patient_id]

        for idx, _ in enumerate(pt_data["Weeks"]):
            dataset["data"][0].append(scan_data)

            data_pt = [
                pt_data["Weeks"][idx],
                pt_data["Age"][0],
                float(sex_to_int[pt_data["Sex"][0]]),
                float(smoking_status_to_int[pt_data["Smoking Status"][0]]),
            ]
            dataset["data"][1].append(data_pt)
            dataset["label"].append(pt_data["FVC"][idx])

        if pt_idx >= 0:
            break

    return dataset


if __name__ == "__main__":
    target_scan_size = (128, 64, 64, 1)
    model = nn_model(scan_size=target_scan_size)

    # TODO: Dynamically find max number of slices
    max_slice_size = 64
    train_dataset = load_dataset(
        scan_data_dir="/projectnb/ece601/F-PuPS/kaggle/data/train",
        csv_dir="/projectnb/ece601/F-PuPS/kaggle/data/", 
        target_scan_size=target_scan_size,
    )
    test_dataset = load_dataset(
        scan_data_dir="/projectnb/ece601/F-PuPS/kaggle/data/test",
        csv_dir="/projectnb/ece601/F-PuPS/kaggle/data/",
        target_scan_size=target_scan_size,
    )

    trainer = ModelTrainer(model=model, run_name="tmp")
    trainer.train(lr=1e-4, train_data=train_dataset, test_data=test_dataset, epochs=1)


from train import correct_dim_size, sex_to_int, smoking_status_to_int, root_mean_squared_error_loss
from tensorflow.keras.models import load_model
from fvc_model.neural_network import nn_model
from loader.loader import load_scan
from tqdm import tqdm

import tensorflow as tf
import scipy.signal
import numpy as np
import os

def fvc_pipeline(patient, categorical_data):
    target_scan_size = (128, 64, 64, 1)
    weight_file = "unfrozen_backbone"
    model = load_model(weight_file, compile=False)
    model.compile(loss=root_mean_squared_error_loss)

    patient_data = {"data": [[], []]}

    scan_data = load_scan(patient)

    slice_dim = scan_data.shape[1]

    # Cropping or padding to appropriate size
    for axis_idx, _ in enumerate(scan_data.shape):
        scan_data = correct_dim_size(scan_data, target_scan_size[axis_idx], axis_idx)

    scan_data = scan_data / 255.
    scan_data = np.expand_dims(scan_data, axis=-1)

    patient_data["data"][0].append(scan_data)

    data_pt = [
        categorical_data["weeks"],
        categorical_data["age"],
        float(sex_to_int[categorical_data["sex"]]),
        float(smoking_status_to_int[categorical_data["smoking_status"]]),
    ]
    patient_data["data"][1].append(data_pt)
    predictions = model.predict(patient_data["data"])
    return predictions[0][0]

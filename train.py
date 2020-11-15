from csv_reader.csv_to_dict import csv_to_dict
from tensorflow.keras.optimizers import Adam
from fvc_model.neural_network import nn_model
from loader.loader import load_scan
from tqdm import tqdm

import tensorflow as tf
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
        self.model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])    
        history = self.model.fit(
            train_data["data"], train_data["label"], 
            batch_size=1, 
            epochs=epochs,
            validation_data=(test_data["data"], test_data["label"])
        )
        return

    def eval(self, test_arr: np.ndarray):
        predictions = model.predict(test_arr)
        return predictions


def load_dataset(scan_data_dir, csv_dir, max_num_slices=None, pad_csv_data=False):
    dataset = {"data": [[], []], "label": []}
    csv_data = csv_to_dict(csv_dir, pad_with_zeros=pad_csv_data)

    if "train" in scan_data_dir:
        csv_data = csv_data[0]
    elif "test" in scan_data_dir:
        csv_data = csv_data[1]

    for pt_idx, patient_id in enumerate(tqdm(os.listdir(scan_data_dir))):
        scan_data = load_scan(os.path.join(scan_data_dir, patient_id))
        slice_dim = scan_data.shape[1]

        if scan_data.shape[0] < max_num_slices:
            scan_data = np.concatenate((scan_data, np.zeros((max_num_slices - scan_data.shape[0], slice_dim, slice_dim))), axis=0)

        scan_data = scan_data / 255.
        scan_data = np.expand_dims(scan_data, axis=0)

        pt_data = csv_data[patient_id]

        for idx, _ in enumerate(pt_data["Weeks"]):
            dataset["data"][0].append(scan_data)

            data_pt = [
                pt_data["Weeks"][idx],
                pt_data["Age"][0],
                float(sex_to_int[pt_data["Sex"][0]]),
                float(smoking_status_to_int[pt_data["Smoking Status"][idx]]),
            ]
            dataset["data"][1].append(data_pt)

            #  data_pt = [scan_data, np.asarray([
            #      pt_data["Weeks"][idx],
            #      pt_data["Age"][0],
            #      float(sex_to_int[pt_data["Sex"][0]]),
            #      float(smoking_status_to_int[pt_data["Smoking Status"][idx]]),
            #  ])]
            #  dataset["data"].append(data_pt)
            dataset["label"].append(pt_data["FVC"][idx])

        if pt_idx > 5:
            break

    #  dataset = np.concatenate(dataset, axis=0)
    return dataset


if __name__ == "__main__":
    model = nn_model()
    train_dataset = load_dataset(
        scan_data_dir="/projectnb/ece601/F-PuPS/kaggle/data/train",
        csv_dir="/projectnb/ece601/F-PuPS/kaggle/data/", 
        max_num_slices=1018,    #CHANGED TO MAX NUMBER OF SLICES FOR ALL PATIENTS
        pad_csv_data=False      #Could change to True if we want to pad the data
    )
    test_dataset = load_dataset(
        scan_data_dir="/projectnb/ece601/F-PuPS/kaggle/data/test",
        csv_dir="/projectnb/ece601/F-PuPS/kaggle/data/",
        max_num_slices=1018,    #CHANGED TO MAX NUMBER OF SLICES FOR ALL PATIENTS
        pad_csv_data=False      ##Could change to True if we want to pad the data. Might be weird for test data since they're all just one week
    )

    trainer = ModelTrainer(model=model, run_name="tmp")
    trainer.train(lr=1e-4, train_data=train_dataset, test_data=test_dataset, epochs=1)


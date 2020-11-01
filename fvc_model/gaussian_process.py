# Handling relative import issue
import sys
sys.path.insert(0, "..")


# Since we are running on a headless server we need to use Agg backend
import matplotlib
matplotlib.use("Agg")


from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from csv_reader.csv_to_dict import csv_to_dict
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np
import random


plt.style.use("ggplot")


smoking_to_int = {
    "Never": 0,
    "Ex-smoker": 1,
    "Currently": 2,
}


sex_to_int = {
    "Male": 0,
    "Female": 1,
}


def plot_data(data, patient_id, fname):
    x = data[patient_id]["Weeks"]
    y = data[patient_id]["FVC"]

    plt.figure()
    plt.scatter(x, y)
    plt.title(patient_id)
    plt.xlabel("Weeks")
    plt.ylabel("FVC")
    plt.savefig(fname)
    return
    

def featurize_train_inputs(input_dict):
    """ 
    Function to featurize a given dataset

    Data includes age, sex, smoking status, weeks relative to scan
    """
    age = input_dict["Age"][0]
    sex = sex_to_int[input_dict["Sex"][0]]
    smoking = smoking_to_int[input_dict["Smoking Status"][0]]
    
    X = []
    y = []
    for idx, week in enumerate(input_dict["Weeks"]):
        X.append([age, sex, smoking, week])
        y.append(input_dict["FVC"][idx])
    return X, y


def featurize_test_inputs(input_dict):
    age = input_dict["Age"][0]
    sex = sex_to_int[input_dict["Sex"][0]]
    smoking = smoking_to_int[input_dict["Smoking Status"][0]]
    week = input_dict["Weeks"][0]
    
    X = [[age, sex, smoking, week]]
    y = [input_dict["FVC"]]
    return X, y


def fit_gaussian_process(X_train, y_train, X_test, y_test):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e-3)) * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e-2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X_train, y_train)

    y_pred, sigma = gp.predict(X_test, return_std=True)
    
    return y_pred


def load_categorical_data(csv_dir):
    train_csv, test_csv = csv_to_dict(csv_dir)
    
    train_patients = list(train_csv.keys())
    test_patients = list(test_csv.keys())

    X_train, y_train = [], []

    for idx, patient_id in enumerate(tqdm(train_patients)):
        #  plot_data(train_csv, patient_id, "{}.png".format(idx))
        X_pt, y_pt = featurize_train_inputs(train_csv[patient_id])
        X_train += X_pt
        y_train += y_pt

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1, 1)

    X_test, y_test = [], []

    for idx, patient_id in enumerate(tqdm(test_patients)):
        X_pt, y_pt = featurize_test_inputs(test_csv[patient_id])
        X_test += X_pt
        y_test += y_pt

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).reshape(-1, 1)
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = load_categorical_data("/projectnb/ece601/F-PuPS/kaggle/data")
    y_pred = fit_gaussian_process(X_train, y_train, X_test, y_test)

    for p, t in zip(y_pred, y_test):
        print(p, t)

    return


if __name__ == "__main__":
    main()


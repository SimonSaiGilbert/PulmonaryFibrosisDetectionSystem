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

## Evaluation Metric Function - Credit to @rohanrao on Kaggle
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    """ Calculates the modified Laplace Log Likelihood score """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    return np.mean(metric)


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
    kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e-2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X_train, y_train)
    return gp


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
    gp = fit_gaussian_process(X_train, y_train, X_test, y_test)
    y_train_pred, train_sigma = gp.predict(X_train, return_std=True)
    y_pred, sigma = gp.predict(X_test, return_std=True)

    y_train = np.squeeze(y_train)
    y_train_pred = np.squeeze(y_train_pred)
    y_test = np.squeeze(y_test)
    y_pred = np.squeeze(y_pred)

    for p, t in zip(y_pred, y_test):
        print(p, t)

    print("Training Laplace Log Likelihood: {}".format(laplace_log_likelihood(y_train, y_train_pred,
        train_sigma*y_train_pred)))
    print("Testing Laplace Log Likelihood: {}".format(laplace_log_likelihood(y_test, y_pred, sigma*y_pred)))
    return


if __name__ == "__main__":
    main()


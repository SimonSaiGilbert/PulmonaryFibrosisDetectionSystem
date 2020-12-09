# Fibrotic Pulmonary Prediction System (F-PuPS)

Jake Hellman  
jhellman@bu.edu  


Simon Gilbert  
simonsai@bu.edu  


Shashank Manjunath  
manjuns@bu.edu  


## Introduction

F-PuPS (Fibrotic Pulmonary Prediction System) is a secure machine learning system that allows clinicians to provide
Idiopathic Pulmonary Fibrosis patients with an accurate prognosis. F-PuPS improves prognosis accuracy by utilizing
machine learning and requires only a single lung scan whereas conventional methods require multiple.

The general flow of our architecture is as follows: 
1. A clinician uploads their patients' scan to the user interface
2. Scan is sent to system storage
3. When the clinician desires, the scan is sent as input to trained neural network which generates prognosis and/or
   segmentation maps of the lung area
4. Prognosis sent to user interface

![](https://github.com/shashankmanjunath/ec601-term-project/blob/SimonSaiGilbert-patch-1/F-PuPS%20Block%20Diagram.png)

## Setting Up The Environment

To create the environment within which to run this project, we use [Anaconda](https://docs.anaconda.com/anaconda/). Once
you have installed Anaconda3, create the environment using the command:

```
conda env create -f environment.yml
```

## Downloading Model Weights

We are currently working to store our segmentation and FVC prediction model weights in a shared location.

## Running the User Interface

The User Interface operates as follows:
- Prompt user to select the patient they'd like to investigate (file picker)
- Provide user with three options once they've selected a patient
    - View the available CT scan data (before processing)
    - View the segmented lung images (after segmentation)
    - Predict the FVC value for the patient (after running model)

As discussed in Sprint 4, we used Tkinter for this implementation. Some screenshots are shown below:

### Select patient
![file picker](https://github.com/shashankmanjunath/ec601-term-project/blob/gui-dev/screenshots/file_picker.png)

### Select an action
![select action](https://github.com/shashankmanjunath/ec601-term-project/blob/gui-dev/screenshots/select_action.png)

### View CT Scan
![view ct scan](https://github.com/shashankmanjunath/ec601-term-project/blob/SimonSaiGilbert-patch-2/view_ct_scan.png)

To start the user interface, run `python interface_guiv2.py` from the Anaconda3 environment created in the previous
step.

## Training the Model

Training the model is not required, but we provide information on the data and training process below.

### Dataset

The Kaggle OSIC Pulmonary Fibrosis dataset was downloaded and is stored on non-backed up disk space on the BU SCC.

Once connected to the SCC, the data can be found here: */projectnb/ECE601/F-PuPS/kaggle/data*.

[The data can additionally be found on Kaggle.](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression)

### Data Loading

Data loading is accomplished by the `load_data` function in `train.py`. We use the given scan parameters
provided in the DICOM file to rescale the given data to Hounsfield Units (HU). This is a unit where pure water is 0 and
high density material (e.g., bone) is 1.

We worked to remove the image border that resulted from the initial segmentation pass though the model. To eliminate the
border, we multiplied the input image (to the model) with the results (from the model). Because the input image contains
a clear region outside of the scan, we were able to threshold in this region and thereby remove it from the output
images. An example of this processing is shown below:

### Initial input to the model:

![initial image](https://github.com/shashankmanjunath/ec601-term-project/blob/jhellman/segmentation/segmentation/processed_data/my_data_test%5B10%5D.png)

### Direct output from the model:

![output from
model](https://github.com/shashankmanjunath/ec601-term-project/blob/jhellman/segmentation/segmentation/output/my_results%5B10%5D.png)

### Results from post-processing to remove background

![post-processed to remove
background](https://github.com/shashankmanjunath/ec601-term-project/blob/jhellman/segmentation/segmentation/output/my_processed_results%5B10%5D.png)

Earlier in the project we made an assumption that all DICOM CT scans would have the same width and height dimensions. We
cropped the images to 512\*512 with this assumption in mind. We know that the number of slices per patient varies. As we
learned during the last sprint, unfortunately this was an invalid assumption and led us to crop out valuable information
for scans larger than 512\*512. We should have noticed this earlier because the row and column values are attributes of
each DICOM file.

In order to mitigate this data size and inconsistency issue, we implemented padding and rescaling operations to create
consistent `(128, 64, 64)` scans. Our data resizing functionality implements the following logic

1. Resample height and width dimensions to `(64, 64)`
2. If depth dimension is less than 128, we zero-pad it to 128
3. If depth dimension is larger than 128, we resample it to 128

### Model Bi-Directional ConvLSTM U-Net

*The code for the Bi-Directional ConvLSTM U-Net is based on the work of [GitHub user
rezazad68](https://github.com/rezazad68/BCDU-Net)*

In the paper referenced [here](https://arxiv.org/pdf/1909.00166.pdf), the authors explain that:
> Among the existing networks, U-Net has been successfully applied on medical image segmentation. In this paper, we
propose an extension of U-Net, Bi-directional ConvLSTM U-Net with Densely connected convolutions (BCDU-Net), for medical
image segmentation, in which we take full advantages of U-Net, bi-directional ConvLSTM (BConvLSTM) and the mechanism of
dense convolutions.

The model is shown by the authors to achieve state of the art performance on retinal blood vessel segmentation, skin
lesion segmentation, and lung nodule segmentation. An image of the network is shown below:

![BCDU-Net block diagram](https://raw.githubusercontent.com/rezazad68/BCDU-Net/master/output_images/bcdunet.png)

We began by replicating the results of the authors, using the SCC to train the model on the annotated dataset used by
the authors. Once complete, we integrated our Kaggle IPF dataset with the model to achieve segmentation results. We do
not have annotated IPF data so there are no score metrics available, however visually the results appear quite
impressive. The results for one of our data volumes is shown below:

![BCDU-Net IPF Results
GIF](https://github.com/JakeHellman/BCDU-Net/blob/master/Lung%20Segmentation/my_sample_results.gif)

### Integration of Categorical Data and Final Model

To integrate the categorical data, we convert categories to discrete integer values. For example, the smoking status
input has three values - "Never", "Ex-smoker", and "Currently". Each was mapped to the integer values 0, 1, and 2,
respectively. We then integrate the segmentation maps with the categorical by first extracting features from the
segmentation map with 3 Convolution and MaxPooling layers before flattening the remaining data and applying a fully
connected layer with 512 hidden units. We apply three fully connected units to the categorical data, with 32, 16, and 8
hidden layers. We then concatenate the segmentation map features (512 values) with the processed categorical features (8
values), and apply two further fully connected layers with 32 and 1 hidden units in order to get a single output value.

### Loss Function

Prediction of FVC, rather than being the more standard classification type problem typically solved using neural network
based techniques, is a regression problem. Instead of deciding what class an image represents, we want our model to
predict an analog FVC value. This requires using a regression-based loss function. Typically, in this situation, Mean
Square Error (MSE) is used as a loss function. However, we decided to focus on Root Mean Square Error (RMSE) for more
standard scaling. This allows us to easily compare our network loss over different loss functions, such as L1 loss, as
well as directly compare loss values to the actual FVC values. We experimented with different loss functions and found
RMSE to lead to the best results.

### Results

Our first experiment consisted of loading in existing weights into our BCDU-Net backbone model, and adding convolutional
layers combined with fully connected layers in order to integrate our categorical data with the segmentation maps
generated from the segmentation maps. For a first experiment, we did not freeze the weights of our segmentation network,
but rather allowed them to be changed by backpropagation. On our 5-user test dataset, we had the following results:

| True FVC | Predicted FVC |
|   ---    |      ---      |
|   3020.0 |     2306.59   |
|   2739.0 |     3048.71   |
|   3294.0 |     3572.02   |
|   2925.0 |     3558.66   |
|   1930.0 |     2593.07   |

This corresponds to an RMSE of 519.37, and a Laplace log likelihood of -12.684. For reference, the winning submission on
Kaggle had a score of -6.83 (a less negative Laplace log likelihood value is better. This is a large RMSE value. Our
next experiments will include freezing our segmentation model backbone, as well as modifying our "regression head" to
improve model performance.

For our next experiment, we froze network backbone weights, and only trained the regression head. In addition to
training much more quickly (training the full model takes ~11.11 hours, while training just the regression head takes
~3.5 hours), this proved to be slightly more accurate, yielding the following results:

| True FVC | Predicted FVC |
|   ---    |      ---      |
|   3020.0 |     2767.97   |
|   2739.0 |     2688.72   |
|   3294.0 |     2681.56   |
|   2925.0 |     2487.90   |
|   1930.0 |     2768.04   |

This corresponds to an RMSE of 437.98, and corresponds to a Laplace Log Likelihood value of -10.707.

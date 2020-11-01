import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import models as M
import numpy as np
import scipy
import matplotlib.pyplot as plt

####################################  Load Data #####################################
folder    = './processed_data/'
te_data   = np.load(folder+'my_data_test.npy')

te_data  = np.expand_dims(te_data, axis=3)

print('Dataset loaded')

####################################  Run Model #####################################
te_data2 = te_data /255.
model = M.BCDU_net_D3(input_size = (512,512,1))
model.summary()
model.load_weights('weight_lung')
predictions = model.predict(te_data2, batch_size=2, verbose=1)

predictions = np.squeeze(predictions)
predictions = np.where(predictions>0.5, 1, 0)

####################################  Save Data #####################################
output_folder = 'output/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

np.save(output_folder+'my_results', predictions)

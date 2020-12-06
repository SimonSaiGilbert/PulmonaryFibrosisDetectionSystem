import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import segmentation.models_segmentation as M
import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras import backend as K

def run_model(input_path, output_path, patient_id):
####################################  Load Data #####################################
    te_data   = np.load(input_path + patient_id + ".npy")
    
    te_data  = np.expand_dims(te_data, axis=3)
    
    print('Dataset loaded')

####################################  Run Model #####################################
    te_data2 = te_data /255.
    model = M.BCDU_net_D3(input_size = (512,512,1))
    #model.summary()             # this line prints out text summarizing the model
    model.load_weights('/projectnb/ece601/F-PuPS/Hellman_working_directory/ec601-term-project/segmentation/weight_lung')
    predictions = model.predict(te_data2, batch_size=2, verbose=1)
    
    predictions = np.squeeze(predictions)
    predictions = np.where(predictions>0.5, 1, 0)

    K.clear_session()

####################################  Save Data #####################################
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    np.save(output_path+patient_id, predictions)

if __name__ == "__main__":
    folder    = './processed_data/'
    run_model(folder,'output/','my_data_test')
    

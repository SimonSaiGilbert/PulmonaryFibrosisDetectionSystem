import os
import numpy as np

####################################  Load Data #####################################
def load_data(input_path_before_model, input_path_after_model, patient_id):
    input_data   = np.load(input_path_before_model+patient_id + '.npy')
    output_data   = np.load(input_path_after_model+patient_id + '.npy')
    
    print('Images to be processed loaded')
    
    return input_data, output_data

##################################  Process Data  ###################################
def process_data(input_data, output_data):
    # create reference value to outside of imaging volume
    empty_reference = input_data[0,0,0]
    
    # make input data one dimensional array for easier processing
    input_data = input_data.reshape(-1)
    
    # create np array for output, initialize as copy of input
    postprocessed_data = np.copy(input_data)
    
    # if adjacent voxels are outside of imaging volume, make 0, else, make 1
    for i, voxel in enumerate(input_data):
        if voxel == empty_reference and input_data[i-1] == empty_reference:
            postprocessed_data[i] = 0
        else:
            postprocessed_data[i] = 1
    
    # reshape data into 512 x 512 images
    postprocessed_data = postprocessed_data.reshape(-1,512,512)
    
    # use the mask generated (0s and 1s) to crop the output data from the model
    postprocessed_data = postprocessed_data * output_data
    
    print('Data postprocessing complete')
    
    return postprocessed_data

####################################  Save Data #####################################
def save_data(postprocessed_data, output_path, patient_id):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    np.save(output_path+patient_id, postprocessed_data)
    
    print('Data saved')

if __name__ == "__main__":
    input_path_before_model    = './processed_data/'
    input_path_after_model     = './output/'
    input_data, output_data = load_data(input_path_before_model,input_path_after_model,'my_data_test')
    postprocessed_data = process_data(input_data, output_data)
    save_data(postprocessed_data,'output/','my_data_test_postprocessed')
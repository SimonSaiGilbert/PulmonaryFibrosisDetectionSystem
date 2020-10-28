import os
import numpy as np

####################################  Load Data #####################################
input_folder    = './processed_data/'
input_data   = np.load(input_folder+'my_data_test.npy')
output_folder = './output/'
output_data   = np.load(output_folder+'my_results.npy')

print('Images to be processed loaded')

##################################  Process Data  ###################################
empty_reference = input_data[0,0,0]
input_data = input_data.reshape(-1)
postprocessed_data = np.copy(input_data)
for i, voxel in enumerate(input_data):
    if voxel == empty_reference and input_data[i-1] == empty_reference:
        postprocessed_data[i] = 0
    else:
        postprocessed_data[i] = 1
postprocessed_data = postprocessed_data.reshape(-1,512,512)

postprocessed_data = postprocessed_data * output_data

print('Data postprocessing complete')

####################################  Save Data #####################################
output_folder = 'output/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

np.save(output_folder+'my_processed_results', postprocessed_data)

print('Data saved')
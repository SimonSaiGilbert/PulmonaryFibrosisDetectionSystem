# pipeline.py
# preprocess_data.py --> evaluate_performance.py --> postprocess_data.py

import segmentation.preprocess_data as pred
import segmentation.evaluate_performance as ev
import segmentation.postprocess_data as postd
import glob


def segmentation_fn(patient):

    input_path, patient_id = patient.rsplit('/',1)
    input_path = input_path + '/'
    
    # Preprocess Data
    output_path = 'preprocessed_data/'
    pred.save_data(pred.load_data(input_path + patient_id + '/'),output_path,patient_id)
    
    # Process Through Network
    input_path = "preprocessed_data/"
    output_path = "output_model/"
    ev.run_model(input_path, output_path, patient_id)
    
    # Postprocess Data
    raw_path = "preprocessed_data/"
    input_path = "output_model/"
    output_path = "output_pipeline/"
    input_data, output_data = postd.load_data(raw_path,input_path,patient_id)
    postprocessed_data = postd.process_data(input_data, output_data)
    postd.save_data(postprocessed_data,output_path,patient_id)

    print("pipeline.py complete")


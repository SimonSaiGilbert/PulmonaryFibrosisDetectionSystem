# pipeline.py
# preprocess_data.py --> evaluate_performance.py --> postprocess_data.py

import preprocess_data as pred
import evaluate_performance as ev
import postprocess_data as postd
import glob

dataset = "test/"        # test/ or train/
data_path = "/projectnb/ece601/F-PuPS/kaggle/data/" + dataset
patients = glob.glob(data_path + "ID*")

for patient in patients:
    input_path, patient_id = patient.rsplit('/',1)
    input_path = input_path + '/'
    
    # Preprocess Data
    output_path = "/projectnb/ece601/F-PuPS/kaggle/preprocessed_data/" + dataset
    pred.save_data(pred.load_data(input_path + patient_id + '/'),output_path,patient_id)
    
    # Process Through Network
    input_path = "/projectnb/ece601/F-PuPS/kaggle/preprocessed_data/" + dataset
    output_path = "/projectnb/ece601/F-PuPS/kaggle/output_model/" + dataset
    ev.run_model(input_path, output_path, patient_id)
    
    # Postprocess Data
    raw_path = "/projectnb/ece601/F-PuPS/kaggle/preprocessed_data/" + dataset
    input_path = "/projectnb/ece601/F-PuPS/kaggle/output_model/" + dataset
    output_path = "/projectnb/ece601/F-PuPS/kaggle/output_pipeline/" + dataset
    input_data, output_data = postd.load_data(raw_path,input_path,patient_id)
    postprocessed_data = postd.process_data(input_data, output_data)
    postd.save_data(postprocessed_data,output_path,patient_id)

print("pipeline.py complete")

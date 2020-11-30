# This program reads in train.csv and test.csv as dictionary objects of the form
# {‘ID00419637202311204720264’: {‘Weeks’: [6.0, 6.0], ‘FVC’: [3020.0, 3020.0], ‘Percent’: [70.18685507, 70.18685507], ‘Age’: [73.0], ‘Sex’: [1.0], ‘Smoking Status’: [1.0, 1.0], ‘ID00421637202311550012437’: …}

# Note that for this program to work, the patients' sex and smoking status must be converted to integers as follows:
# Never smoked = 0, Ex-smoker = 1, Currently smokes = 2
# Female = 0, Male = 1

import csv
import os

def csv_to_dict(csv_directory, pad_with_zeros=False):
	csv_train_dict = {}
	csv_test_dict = {}
	max_length = 10 #This is the maximum number of weeks of data for any patient in the training set.
	padding_keys = ["Weeks", "FVC", "Percent", "Smoking Status"] #These are the rows that will be padded if pad_with_zeros == True

	with open(os.path.join(csv_directory, "test.csv"), newline = '') as csv_test_file:
		test_data_reader = csv.reader(csv_test_file, delimiter=' ', quotechar='|')
		for row in test_data_reader:
			row = row[0].split(',')

			if row[0] == 'Patient':
				pass

			else:
				if row[0] not in csv_test_dict.keys():
					csv_test_dict[row[0]] = {}
					csv_test_dict[row[0]]['Weeks'] = []
					csv_test_dict[row[0]]['FVC'] = []
					csv_test_dict[row[0]]['Percent'] = []
					csv_test_dict[row[0]]['Age'] = [float(row[4])]
					csv_test_dict[row[0]]['Sex'] = [str(row[5])]
					csv_test_dict[row[0]]['Smoking Status'] = []

				csv_test_dict[row[0]]['Weeks'].append(float(row[1]))
				csv_test_dict[row[0]]['FVC'].append(float(row[2]))
				csv_test_dict[row[0]]['Percent'].append(float(row[3]))
				csv_test_dict[row[0]]['Smoking Status'].append(str(row[6]))

			
	with open(os.path.join(csv_directory, "train.csv"), newline = '') as csv_train_file:
		train_data_reader = csv.reader(csv_train_file, delimiter=' ', quotechar='|')
		for row in train_data_reader:
			row = row[0].split(',')

			if row[0] == 'Patient':
				pass

			else:
				if row[0] not in csv_train_dict.keys():
					csv_train_dict[row[0]] = {}
					csv_train_dict[row[0]]['Weeks'] = []
					csv_train_dict[row[0]]['FVC'] = []
					csv_train_dict[row[0]]['Percent'] = []
					csv_train_dict[row[0]]['Age'] = [float(row[4])]
					csv_train_dict[row[0]]['Sex'] = [str(row[5])]
					csv_train_dict[row[0]]['Smoking Status'] = []

				csv_train_dict[row[0]]['Weeks'].append(float(row[1]))
				csv_train_dict[row[0]]['FVC'].append(float(row[2]))
				csv_train_dict[row[0]]['Percent'].append(float(row[3]))
				csv_train_dict[row[0]]['Smoking Status'].append(str(row[6]))

	if pad_with_zeros == True:
		#Pads values in test dictionary
		for ID in csv_test_dict.keys():
			for key in padding_keys:
				while len(csv_test_dict[ID][key]) < max_length:
					csv_test_dict[ID][key].append(0)

		#Pads values in train dictionary
		for ID in csv_train_dict.keys():
			for key in padding_keys:
				while len(csv_train_dict[ID][key]) < max_length:
					csv_train_dict[ID][key].append(0.0)


	return csv_train_dict, csv_test_dict

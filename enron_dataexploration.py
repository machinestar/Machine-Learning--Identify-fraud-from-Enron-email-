#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### total number of data points
print 'the total number of data points is:', len(data_dict)

### number of POI 
number_of_poi = 0
for person in data_dict:
	datapoint = data_dict[person]
	if datapoint["poi"]:
		number_of_poi = number_of_poi + 1

print 'there are %d poi in this dataset' %number_of_poi
	
####     outlier identify ######## 
for person in data_dict:
	datapoint = data_dict[person]
	salary = datapoint["salary"]
	total_stock_value = datapoint["total_stock_value"]
	matplotlib.pyplot.scatter(salary, total_stock_value)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("total_stock_value")
matplotlib.pyplot.show()

for person in data_dict:
	datapoint = data_dict[person]
	salary = datapoint["salary"]
	bonus = datapoint["bonus"]
	matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### There are one point far beyond the normal range, decided to detect which persion is associated with this point

outliersalary = 0
outlierbonus = 0
outlierstock = 0

for key, value in data_dict.items():
	if value['salary'] > outliersalary and value['salary'] != 'NaN':
		outliersalary = value['salary']
		salarykey = key
	if value['bonus'] > outlierbonus and value['bonus'] != 'NaN':
		outlierbonus = value['bonus']
		bonuskey = key
	if value['total_stock_value'] > outlierstock and value['total_stock_value'] != 'NaN':
		outlierstock = value['total_stock_value']
		stockkey = key
print salarykey, bonuskey, stockkey, key

### Missing Values distribution

feature_NaN_value = dict()
for person in data_dict:
	datapoint = data_dict[person]
	for feature in datapoint:
		if datapoint[feature] == 'NaN':
			feature_NaN_value[feature] = feature_NaN_value.get(feature, 0) + 1

for key, value in feature_NaN_value.items():
	print key, value



	



		
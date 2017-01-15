#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt

### create function to generate fraction of emails sent to and from poi
def computeFraction(poi_messages, all_messages):
	fraction = 0
	if poi_messages !="NaN" and all_messages !="NaN":
		fraction = float(poi_messages)/float(all_messages)
	return fraction



with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### delete outlier
del data_dict['TOTAL']	

### add new feature to feature list
for name in data_dict:
	data_point = data_dict[name]
	from_poi_to_this_person = data_point["from_poi_to_this_person"]
	to_messages = data_point["to_messages"]
	fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
	data_point["fraction_from_poi"] = fraction_from_poi

	from_this_person_to_poi = data_point["from_this_person_to_poi"]
	from_messages = data_point["from_messages"]
	fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
	data_point["fraction_to_poi"] = fraction_to_poi
	
	shared_receipt_with_poi = data_point["shared_receipt_with_poi"]
	to_messages_1 = data_point["to_messages"]
	fraction_shared_poi = computeFraction(shared_receipt_with_poi, to_messages_1)
	data_point["fraction_shared_poi"] = fraction_shared_poi



features_list_all = ['poi', 'salary','total_payments', 'long_term_incentive', 'bonus', 'restricted_stock', 'total_stock_value','shared_receipt_with_poi','from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'expenses', 'fraction_from_poi', 'fraction_to_poi', 'fraction_shared_poi']

### Load the dictionary containing the dataset
data_all = featureFormat(data_dict, features_list_all)

### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing
labels_all, features_all = targetFeatureSplit(data_all)


### test different numbers of features selected by RFE
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostClassifier

ada_num_dict = dict() ## dump the F1 scores into dict.

for n in range(16):
	print 'the number of selected features is:', n+1
	estimator = AdaBoostClassifier()
	selector_adaboost = RFE(estimator, step=1,n_features_to_select= n+1)
	selector_adaboost.fit(features_all, labels_all)

	feature_selected_adaboost = [features_list_all[i+1] for i in selector_adaboost.get_support(indices=True)]
	print 'The Features Selected for adaboost:'
	print feature_selected_adaboost

	features_list_adaboost = ['poi'] + feature_selected_adaboost

	data_adaboost = featureFormat(data_dict, features_list_adaboost)
	labels_adaboost, features_adaboost = targetFeatureSplit(data_adaboost)

	from sklearn import grid_search
	from sklearn.cross_validation import StratifiedShuffleSplit

	cv_adaboost = StratifiedShuffleSplit(labels_adaboost, n_iter = 100, random_state = 42)
	parameters_adaboost = {'n_estimators': [10, 20, 30, 40, 60, 100], 'algorithm' : ['SAMME', 'SAMME.R'], 'random_state': [42]}
	adaboost = AdaBoostClassifier()
	clf_adaboost = grid_search.GridSearchCV(adaboost, parameters_adaboost, scoring = 'f1', cv=cv_adaboost)

	clf_adaboost.fit(features_adaboost, labels_adaboost)
	print 'best parameter for adaboost is', clf_adaboost.best_params_
	
	
	ada_num_dict[n+1] = clf_adaboost.best_score_

for key, value in ada_num_dict.items():
	print key, value 


tree_num_dict = dict()

for n in range(16):
	from sklearn import tree
	estimator = tree.DecisionTreeClassifier()
	print 'the number of selected features is:', n+1
	selector_tree = RFE(estimator, step=1,n_features_to_select= n+1)
	selector_tree.fit(features_all, labels_all)

	feature_selected_tree = [features_list_all[i+1] for i in selector_tree.get_support(indices=True)]
	print 'The Features Selected for tree:'
	print feature_selected_tree

	features_list_tree = ['poi'] + feature_selected_tree

	data_tree = featureFormat(data_dict, features_list_tree)
	labels_tree, features_tree = targetFeatureSplit(data_tree)

	from sklearn import grid_search
	from sklearn.cross_validation import StratifiedShuffleSplit

	cv_tree = StratifiedShuffleSplit(labels_tree, n_iter = 100, random_state = 42)
	parameters_tree = {'max_features': ['auto', 'sqrt', 'log2'], 'min_samples_split': [1,2,3], 'random_state': [42], 'class_weight': [{1:19, 0:1}]}
	tree = tree.DecisionTreeClassifier()
	clf_tree = grid_search.GridSearchCV(tree, parameters_tree, scoring = 'f1', cv=cv_tree)

	clf_tree.fit(features_tree, labels_tree)
	print 'best parameter for tree is', clf_tree.best_params_
	print 'the best score is:', clf_tree.best_score_
	
	tree_num_dict[n+1] = clf_tree.best_score_

for key, value in tree_num_dict.items():
	print key, value
	
from sklearn.ensemble import RandomForestClassifier
random_num_dict = dict()

for n in range(16):
	print 'the number of selected features is:', n+1	
	estimator = RandomForestClassifier(random_state = 0)
	selector_randomforest = RFE(estimator, step=1,n_features_to_select= n+1)
	selector_randomforest.fit(features_all, labels_all)

	feature_selected_randomforest = [features_list_all[i+1] for i in selector_randomforest.get_support(indices=True)]
	print 'The Features Selected for randomforest:'
	print feature_selected_randomforest

	features_list_randomforest = ['poi'] + feature_selected_randomforest
	
	from sklearn import grid_search
	from sklearn.cross_validation import StratifiedShuffleSplit

	data_randomforest = featureFormat(data_dict, features_list_randomforest)
	labels_randomforest, features_randomforest = targetFeatureSplit(data_randomforest)

	cv_randomforest = StratifiedShuffleSplit(labels_randomforest, n_iter = 100, random_state = 42)
	from sklearn.ensemble import RandomForestClassifier
	parameters_random = {'n_estimators': [10, 20, 30, 40, 60, 100], 'max_features': ['auto', 'sqrt', 'log2'], 'min_samples_split': [1,2,3], 'random_state': [42]}
	randomtree = RandomForestClassifier()
	clf_random = grid_search.GridSearchCV(randomtree, parameters_random, scoring = 'f1', cv=cv_randomforest)

	clf_random.fit(features_randomforest, labels_randomforest)
	print 'best parameter for randomforest is:', clf_random.best_params_ 
	print 'the best score is:', clf_random.best_score_

	random_num_dict[n+1] = clf_random.best_score_

for key, value in random_num_dict.items():
	print key, value


### best F1 score without new features: 'fraction_from_poi', 'fraction_to_poi', 'fraction_shared_poi'

features_list_no_new = ['poi', 'salary','total_payments', 'long_term_incentive', 'bonus', 'restricted_stock', 'total_stock_value','shared_receipt_with_poi','from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'expenses']
data_no_new = featureFormat(data_dict, features_list_no_new)
labels_all, features_no_new = targetFeatureSplit(data_no_new)

adaboost_no_new_feature = dict()
for n in range(13):
	print 'the number of selected features is:', n+1
	estimator = AdaBoostClassifier()
	selector_adaboost = RFE(estimator, step=1,n_features_to_select= n+1)
	selector_adaboost.fit(features_no_new, labels_all)

	feature_selected_adaboost = [features_list_no_new[i+1] for i in selector_adaboost.get_support(indices=True)]
	print 'The Features Selected for adaboost:'
	print feature_selected_adaboost


	features_list_adaboost = ['poi'] + feature_selected_adaboost
	
	data_adaboost = featureFormat(data_dict, features_list_adaboost)
	labels_adaboost, features_adaboost = targetFeatureSplit(data_adaboost)

	from sklearn import grid_search
	from sklearn.cross_validation import StratifiedShuffleSplit

	cv_adaboost = StratifiedShuffleSplit(labels_adaboost, n_iter = 100, random_state = 42)
	parameters_adaboost = {'n_estimators': [10, 20, 30, 40, 60, 100], 'algorithm' : ['SAMME', 'SAMME.R'], 'random_state': [42]}
	adaboost = AdaBoostClassifier()
	clf_adaboost = grid_search.GridSearchCV(adaboost, parameters_adaboost, scoring = 'f1', cv=cv_adaboost)

	clf_adaboost.fit(features_adaboost, labels_adaboost)
	print 'best parameter for adaboost is', clf_adaboost.best_params_
	print 'the best score without new email features is:', clf_adaboost.best_score_
	
	adaboost_no_new_feature[n+1] = clf_adaboost.best_score_

### plot F1 scores for different combinations of classifiers and feature lists
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Validation Score F1")
plt.plot(range(1, 17), ada_num_dict.values(), 'b-', label = "Adaboost")
plt.plot(range(1, 17), tree_num_dict.values(), 'r-', label = "Tree Classifier")
plt.plot(range(1, 17), random_num_dict.values(), 'g-', label = "Random Forest")
plt.legend(['Adaboost', 'Tree Classifier', 'Random Forest'], loc='upper right', fontsize = 'x-small')
plt.show()

### compare F1 score of Adaboost with and without new email features

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Validation Score F1")
plt.plot(range(1, 17), ada_num_dict.values(), 'b-', label = "Adaboost_with_new_features")
plt.plot(range(1, 14), adaboost_no_new_feature.values(), 'r-', label = "Adaboost_without_new_features")
plt.legend(['Adaboost_with_new_features', 'Adaboost_without_new_features',], loc='upper right', fontsize = 'x-small')
plt.show()


### feature importance of Adaboost

clf = AdaBoostClassifier(n_estimators = 60, random_state = 0, algorithm = 'SAMME')
features_list = ['poi', 'shared_receipt_with_poi', 'exercised_stock_options', 'other', 'expenses', 'fraction_to_poi']
data_adaboost = featureFormat(data_dict, features_list)
labels_adaboost, features_adaboost = targetFeatureSplit(data_adaboost)
clf.fit(features_adaboost, labels_adaboost)
print clf.feature_importances_  



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
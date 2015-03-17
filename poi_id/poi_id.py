#!/usr/bin/python
import os
import sys
import pickle
import operator
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit
from parse_out_email_text import parseOutText

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

financial_features_list = ["salary", "deferral_payments", "total_payments", "loan_advances", "bonus", "restricted_stock_deferred",
                           "deferred_income", "total_stock_value", "expenses", "exercised_stock_options", "other", "long_term_incentive", "restricted_stock", "director_fees"]
email_features_list = ["to_messages", "from_poi_to_this_person",
                       "from_messages", "from_this_person_to_poi", "shared_receipt_with_poi"]
target_label = ["poi"]

features_list = target_label + financial_features_list + email_features_list
financial_features_list = target_label + financial_features_list
email_features_list = target_label + email_features_list

data_dict = pickle.load(open("dataset.pkl", "r"))

# Removed outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# Generating a list of emails from those which are available in the
# data
emailers = []
text_file = os.listdir("../final_project/emails_by_address/")
for key in data_dict.keys():
    name = "from_" + data_dict[key]["email_address"] + ".txt"
    if name not in text_file:
        del data_dict[key]
    else:
        emailers.append(key)
emailers.sort()

my_dataset = data_dict

# These two lines extract the features specified in features_list
# and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

word_features = []

# Processing email data to extract word features
for i in range(len(emailers)):
    email = data_dict[emailers[i]]["email_address"]
    person = "../final_project/emails_by_address/from_" + email + ".txt"
    paths = open(person, "r")
    temp_string = ""
    for path in paths:
        path = os.path.join('..', path[:-1])
        print path
        message = open(path, "r")
        parsed_text = parseOutText(message)
        temp_string += parsed_text
        message.close()
    word_features.append(temp_string)
    paths.close()
print "emails processed"

# Split into labels and features
labels, features = targetFeatureSplit(data)

accuracy = []
precision = []
recall = []
f1 = []

vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
scaler = MinMaxScaler()
selector = SelectPercentile(f_classif, percentile=10)
skf = StratifiedKFold(labels, n_folds=3)

for train_idx, test_idx in skf:
    features_train = []
    features_test = []
    word_features_train = []
    word_features_test = []
    labels_train = []
    labels_test = []

    for ii in train_idx:
        features_train.append(features[ii])
        word_features_train.append(word_features[ii])
        labels_train.append(labels[ii])
    for jj in test_idx:
        features_test.append(features[jj])
        word_features_test.append(word_features[jj])
        labels_test.append(labels[jj])

    word_features_train = vectorizer.fit_transform(
        word_features_train).toarray()
    features_train = np.hstack((features_train, word_features_train))
    word_features_test = vectorizer.transform(word_features_test).toarray()
    features_test = np.hstack((features_test, word_features_test))

    features_train = scaler.fit_transform(features_train)
    features_train = selector.fit_transform(features_train, labels_train)
    features_test = scaler.transform(features_test)
    features_test = selector.transform(features_test)

    ada = AdaBoostClassifier()
    parameters = {'n_estimators': [1, 100], 'random_state': [1, 50]}
    clf = GridSearchCV(ada, parameters)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print "Accuracy: ", clf.score(features_test, labels_test)
    print "Precision: ", precision_score(labels_test, pred)
    print "Recall: ", recall_score(labels_test, pred)
    print "F1_Score: ", f1_score(labels_test, pred)

    accuracy.append(clf.score(features_test, labels_test))
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))
    f1.append(f1_score(labels_test, pred))

print "Average accuracy: ", sum(accuracy) / 3
print "Average precision: ", sum(precision) / 3
print "Average recall: ", sum(recall) / 3
print "Average f1_score: ", sum(f1) / 3

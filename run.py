import numpy as np

import dataset
import classifier
import mmfa


print("\n*-------- Experiment Yale --------*")

print("loading data ...")
train_data, train_labels, test_data, test_labels = dataset.Yale(0, 1102)

# dimension reduction
print("performing MMFA ...")
mapping = mmfa.MMFA(train_data, train_labels, k_1=18, k_2=10, binary_weight=False)
train_data_low = np.dot(train_data, mapping)
test_data_low = np.dot(test_data, mapping)

_NN_accuracy, _NN_precision, _NN_recall, _NN_f_score = classifier.knn_score(
    train_data_low, train_labels, test_data_low, test_labels, 1
)
print("Accuracy:", _NN_accuracy)


print("\n*-------- Experiment AR --------*")

print("loading data ...")
train_data, train_labels, test_data, test_labels = dataset.AR(0, 700)

# dimension reduction
print("performing MMFA ...")
mapping = mmfa.MMFA(train_data, train_labels, k_1=3, k_2=6, binary_weight=False)
train_data_low = np.dot(train_data, mapping)
test_data_low = np.dot(test_data, mapping)

## NN
_NN_accuracy, _NN_precision, _NN_recall, _NN_f_score = classifier.knn_score(
    train_data_low, train_labels, test_data_low, test_labels, 1
)
print("Accuracy:", _NN_accuracy)

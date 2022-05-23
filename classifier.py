import sklearn.metrics as SM
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier

def knn_score(train_data, train_label, test_data, test_label, knn_k=1):
    knn = neighbors.KNeighborsClassifier(n_neighbors = knn_k)
    knn.fit(train_data, train_label)
    pre_label = knn.predict(test_data)
    
    accuracy = SM.accuracy_score(test_label, pre_label)
    precision = SM.precision_score(test_label, pre_label, average = 'weighted')
    recall = SM.recall_score(test_label, pre_label, average = 'weighted')
    f_score = SM.f1_score(test_label, pre_label, average = 'weighted')
    
    return accuracy, precision, recall, f_score

def SVM(train_data, train_label, test_data, test_label):
    clf = svm.LinearSVC()
    clf.fit(train_data, train_label)
    pre_label = clf.predict(test_data)
    
    accuracy = SM.accuracy_score(test_label, pre_label)
    precision = SM.precision_score(test_label, pre_label, average = 'weighted')
    recall = SM.recall_score(test_label, pre_label, average = 'weighted')
    f_score = SM.f1_score(test_label, pre_label, average = 'weighted')
    
    return accuracy, precision, recall, f_score

def MLP(train_data, train_label, test_data, test_label):
    mlp = MLPClassifier()
    mlp.fit(train_data, train_label)
    pre_label = mlp.predict(test_data)
    
    accuracy = SM.accuracy_score(test_label, pre_label)
    precision = SM.precision_score(test_label, pre_label, average = 'weighted')
    recall = SM.recall_score(test_label, pre_label, average = 'weighted')
    f_score = SM.f1_score(test_label, pre_label, average = 'weighted')
    
    return accuracy, precision, recall, f_score


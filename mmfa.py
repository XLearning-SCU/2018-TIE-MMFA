import os
import math

import numpy as np
import scipy.linalg as slg


def PCA(data, dimensions):
    '''
    data is the original data. m*n(m=samples, n=dimensions)
    '''
    # making data zero-means
    average = np.mean(data,0)
    data = np.mat(data-average)
    
    covariance = np.dot(data.T, data)
    eig_var, eig_vec = np.linalg.eig(covariance)
    sort_eig = np.argsort(-eig_var)
    sort_eig = sort_eig[:dimensions]
    principal_vec = eig_vec[:,sort_eig]
    low_data = np.dot(data, principal_vec)
    
    return low_data, principal_vec, average


def MMFA(data, label, k_1, k_2, pca_op=1, binary_weight=True):
    # Number of samples N; Dimensions D; Clases c
    data= np.matrix(data)
    [N, Dim] = data.shape
    classes = np.unique(label)
    c = len(classes)
    
    #PCA 
    if(pca_op==1):
        data_pca, mapping_pca, average= PCA(data, N - c)
    else:
        data_pca = data
    
    # neighbors
    data_tmp = np.sum(np.multiply(data_pca, data_pca), axis=1)
    distance=np.mat(data_tmp + data_tmp.T - 2*data_pca*data_pca.T)
    
    # the all neibors N
    neighbors = np.argsort(distance,axis=1)
    neighbors = neighbors[:, 1:]
    
    W = np.zeros((N, N)) 
    W_ = np.zeros((N, N))
    D = np.zeros((N, N)) 
    D_ = np.zeros((N, N))
    
    #  W 
    for i in range(N):
        K_1 = 0
        for j in neighbors[i].A[0]:
            if (label[j]==label[i]):
                if(K_1 < k_1):
                    if binary_weight:
                        W[i, j] = 1
                        W[j, i] = 1
                    else:
                        W[i, j] = distance[i, j]
                        W[j, i] = W[i, j]  
                    K_1 += 1
    
    #  W_
    for c in classes:
        class_c = np.where(label == c)[0]
        for _c in classes:
            if(c != _c):
                _class_c = np.where(label == _c)[0]
                distance_c = distance[class_c, :]
                distance_c = distance_c[:, _class_c]
                arg_distance_c = np.dstack(np.unravel_index(np.argsort(distance_c.ravel()), distance_c.shape))
                arg_distance_c = arg_distance_c[0, :k_2]
                for i in arg_distance_c:
                    x = class_c[i[0]]
                    y = _class_c[i[1]]
                    if binary_weight:
                        W_[x, y] = 1
                        W_[y, x] = 1
                    else:
                        W_[x, y] = distance[x, y]
                        W_[y, x] = W_[x, y]
                        
    for i in range(N):
        D[i, i] = np.sum(W[i, :])
        D_[i, i] = np.sum(W_[i, :])

    L = D-W
    L_ = D_-W_
    X_1 = np.dot(np.dot(data_pca.T, L), data_pca)
    X_2 = np.dot(np.dot(data_pca.T, L_), data_pca)
    #eigenvalues, eigenvectors = np.linalg.eig(np.dot(X_1.I, X_2))
    eigenvalues, eigenvectors = slg.eig(X_2, X_1)
    sort_eig = np.argsort(-eigenvalues)
    mapping = eigenvectors[:, sort_eig[:(N-k_2*c)]]
    
    if(pca_op==1):
        return np.dot(mapping_pca, mapping).astype('float')
    else:
        return mapping.astype('float')
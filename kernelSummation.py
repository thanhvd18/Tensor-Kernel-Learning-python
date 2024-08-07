import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorly.decomposition import parafac
import tensorly as tl


from KL import KL
from utils import *
  
class KernelSummation:
    """

    """
    def __init__(self,n_modalities=None, n_class=2,n_tree=500):
        self.n_class = n_class
        self.X_train = None
        self.n_tree = n_tree
        self.KL_list = []
        self.n_modalities = n_modalities
        # self.X_train_list = []
        # self.K_list = None

    def fit(self, X_train_list, y_train,val_size=0):
        self.KL_list = []
        self.X_train_list = []
        self.n_modalities = len(X_train_list)
        for X_train in X_train_list:
            # train one model for each modality to create kernel matrix K
            self.X_train_list.append(X_train)
            KL_i = KL() # kernel learning
            KL_i.fit(X_train,y_train,val_size=val_size)
            self.KL_list.append(KL_i)
            # K_train = KL_i.get_kernel_matrix(X_train)
            # self.K_train_list.append(K_train) 
        self.y_train = y_train
        
    def predict(self,X_test_list_):
        X_test_list = X_test_list_.copy()
        self.K_list = []
        K_ten = []
        index_train = np.array(range(self.X_train_list[0].shape[0]))
        index_test =  len(index_train) + np.array(range(X_test_list[0].shape[0]))
        for i,X_test in enumerate(X_test_list):
            X = np.vstack([self.X_train_list[i], X_test])
            K = self.KL_list[i].get_kernel_matrix(X)
            K_ten.append(K)
        K_ten = np.array(K_ten)
    
        
        K_combine = np.sum(K_ten,axis=0)
        # K_combine_summation =  np.zeros_like(K_combine)
        
        K_combine = K_combine/ np.sqrt(np.linalg.norm(K_combine))
        K_train = K_combine[np.ix_(index_train,index_train)]
        K_test = K_combine[np.ix_(index_test,index_train)]

        clf = SVC(kernel='precomputed') 
        clf.fit(K_train, self.y_train)
        y_pred = clf.predict(K_test)
        y_pred_summation = y_pred.copy()
    
        y_combine = np.hstack((self.y_train, y_pred))
        idx_sorted = np.argsort(y_combine)
        inverse_idx = np.argsort(idx_sorted)
        
        K_ten_sorted  = np.zeros_like(K_ten)
        for i in range(4):
            K_ten_sorted[i,:,:] = np.squeeze(K_ten[i,:,:])[np.ix_(idx_sorted,idx_sorted)]
        # plt.imshow(K_ten_shuffle[0,:,:])
        K_combine_summation = np.sum(K_ten_sorted,axis=0)
        self.K = K_combine_summation
        return y_pred
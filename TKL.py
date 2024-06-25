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
  
class TKL:
    """
        Tensor kernel learning
    """
    def __init__(self,n_modalities=None, n_class=2,kNN=3, iterations=5,n_tree=500):
        self.n_class = n_class
        self.kNN = kNN
        self.iterations = iterations    
        self.K_train = None
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
        
             
    def predict(self,X_test_list):
        self.K_list = []
        K_ten = []
        index_train = np.array(range(self.X_train_list[0].shape[0]))
        index_test =  len(index_train) + np.array(range(X_test_list[0].shape[0]))
        for i,X_test in enumerate(X_test_list):
            X = np.vstack([self.X_train_list[i], X_test])
            K = self.KL_list[i].get_kernel_matrix(X)
            K_ten.append(K)
        K_ten = np.array(K_ten)
        K_ten = parafac(K_ten, rank=2, tol=1e-9,n_iter_max=1000,init='random')
        K_ten = tl.cp_to_tensor(K_ten)
        K_ten_dp = K_ten ## 
        ## check this step
        # K_ten_dp = cross_diffusion(K_ten, iterations=self.iterations, kNN=self.kNN)
        
        K_combine = np.sum(K_ten_dp,axis=0)
        # K_combine = np.squeeze(K_ten[1:,:])
        
        K_combine = K_combine/ np.sqrt(np.linalg.norm(K_combine))
        K_train = K_combine[np.ix_(index_train,index_train)]
        K_test = K_combine[np.ix_(index_test,index_train)]

        clf = SVC(kernel='precomputed') 
        clf.fit(K_train, self.y_train)
        y_pred = clf.predict(K_test)
        return y_pred
    def predict_(self,X_test_list):
        self.K_list = []
        K_ten = []
        index_train = np.array(range(self.X_train_list[0].shape[0]))
        index_test =  len(index_train) + np.array(range(X_test_list[0].shape[0]))
        for i,X_test in enumerate(X_test_list):
            X = np.vstack([self.X_train_list[i], X_test])
            K = self.KL_list[i].get_kernel_matrix(X)
            K_ten.append(K)
        K_ten_old = K_ten
        K_ten = np.array(K_ten)
        K_ten = parafac(K_ten, rank=2, tol=1e-9,n_iter_max=1000,init='random')
        K_ten = tl.cp_to_tensor(K_ten)
        # K_ten_dp = cross_diffusion(K_ten, iterations=self.iterations, kNN=self.kNN)
        
        # K_combine = np.sum(K_ten_dp,axis=0)
        # # K_combine = np.squeeze(K_ten[1:,:])
        
        # K_combine = K_combine/ np.sqrt(np.linalg.norm(K_combine))
        # K_train = K_combine[np.ix_(index_train,index_train)]
        # K_test = K_combine[np.ix_(index_test,index_train)]

        # clf = SVC(kernel='precomputed') 
        # clf.fit(K_train, self.y_train)
        # y_pred = clf.predict(K_test)
        return 0,K_ten,K_ten_old,index_train,index_test
    
    def predict_summation(self,X_test_list):
        self.K_list = []
        K_ten = []
        index_train = np.array(range(self.X_train_list[0].shape[0]))
        index_test =  len(index_train) + np.array(range(X_test_list[0].shape[0]))
        for i,X_test in enumerate(X_test_list):
            X = np.vstack([self.X_train_list[i], X_test])
            K = self.KL_list[i].get_kernel_matrix(X)
            K_ten.append(K)
        K_ten = np.array(K_ten)
        # K_ten = parafac(K_ten, rank=2, tol=1e-9,n_iter_max=1000,init='random')
        # K_ten = tl.cp_to_tensor(K_ten)
        # K_ten_dp = K_ten ## 
        # ## check this step
        # K_ten_dp = cross_diffusion(K_ten, iterations=self.iterations, kNN=self.kNN)
        
        K_combine = np.sum(K_ten,axis=0)
        # K_combine = np.squeeze(K_ten[1:,:])
        
        K_combine = K_combine/ np.sqrt(np.linalg.norm(K_combine))
        K_train = K_combine[np.ix_(index_train,index_train)]
        K_test = K_combine[np.ix_(index_test,index_train)]

        clf = SVC(kernel='precomputed') 
        clf.fit(K_train, self.y_train)
        y_pred = clf.predict(K_test)
        return y_pred
    def predict_summation_TKL(self,X_test_list):
        self.K_list = []
        K_ten = []
        index_train = np.array(range(self.X_train_list[0].shape[0]))
        index_test =  len(index_train) + np.array(range(X_test_list[0].shape[0]))
        for i,X_test in enumerate(X_test_list):
            X = np.vstack([self.X_train_list[i], X_test])
            K = self.KL_list[i].get_kernel_matrix(X)
            K_ten.append(K)
        K_ten = np.array(K_ten)
        K_ten = parafac(K_ten, rank=2, tol=1e-9,n_iter_max=1000,init='random')
        K_ten = tl.cp_to_tensor(K_ten)
        # K_ten_dp = K_ten ## 
        # ## check this step
        # K_ten_dp = cross_diffusion(K_ten, iterations=self.iterations, kNN=self.kNN)
        
        K_combine = np.sum(K_ten,axis=0)
        # K_combine = np.squeeze(K_ten[1:,:])
        
        K_combine = K_combine/ np.sqrt(np.linalg.norm(K_combine))
        K_train = K_combine[np.ix_(index_train,index_train)]
        K_test = K_combine[np.ix_(index_test,index_train)]

        clf = SVC(kernel='precomputed') 
        clf.fit(K_train, self.y_train)
        y_pred = clf.predict(K_test)
        return y_pred
    
    def predict_summation_DP(self,X_test_list):
        self.K_list = []
        K_ten = []
        index_train = np.array(range(self.X_train_list[0].shape[0]))
        index_test =  len(index_train) + np.array(range(X_test_list[0].shape[0]))
        for i,X_test in enumerate(X_test_list):
            X = np.vstack([self.X_train_list[i], X_test])
            K = self.KL_list[i].get_kernel_matrix(X)
            K_ten.append(K)
        K_ten = np.array(K_ten)
        # K_ten = parafac(K_ten, rank=2, tol=1e-9,n_iter_max=1000,init='random')
        # K_ten = tl.cp_to_tensor(K_ten)
        # K_ten_dp = K_ten ## 
        # ## check this step
        K_ten = cross_diffusion(K_ten, iterations=self.iterations, kNN=self.kNN)
        
        K_combine = np.sum(K_ten,axis=0)
        # K_combine = np.squeeze(K_ten[1:,:])
        
        K_combine = K_combine/ np.sqrt(np.linalg.norm(K_combine))
        K_train = K_combine[np.ix_(index_train,index_train)]
        K_test = K_combine[np.ix_(index_test,index_train)]

        clf = SVC(kernel='precomputed') 
        clf.fit(K_train, self.y_train)
        y_pred = clf.predict(K_test)
        return y_pred
    
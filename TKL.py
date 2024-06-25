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

    def fit(self, X_train_list, y_train):
        self.KL_list = []
        self.X_train_list = []
        self.n_modalities = len(X_train_list)
        for X_train in X_train_list:
            self.X_train_list.append(X_train)
            KL_i = KL()
            KL_i.fit(X_train,y_train)
            self.KL_list.append(KL_i)
            
        self.y_train = y_train
        
             
    def predict_individual_multimodal(self, X_test_list):
        result = []
        self.K_list = []
        for i,X_test in enumerate(X_test_list):
            X = np.vstack([self.X_train_list[i], X_test])
            index_train = np.array(range(self.X_train_list[i].shape[0]))
            index_test =  len(index_train) + np.array(range(X_test.shape[0]))
            # ic(i, len())
            K = self.KL_list[i].get_kernel_matrix(X)
            self.K_list.append( K)
            
            K_train = K[np.ix_(index_train,index_train)]
            K_test = K[np.ix_(index_test,index_train)]
            
            clf = SVC(kernel='precomputed') 
            clf.fit(K_train, self.y_train)
            y_pred = clf.predict(K_test)
            result.append({f"modality": i, "y": y_pred})
        return result
        
    def TKL_train(self):
        
        K_X = np.copy(np.array(self.K_list))
        K_ten = parafac(K_X, rank=2, tol=1e-9,n_iter_max=1000,init='random')
        
        K_ten = tl.cp_to_tensor(K_ten)

        self.K_ten_dp = cross_diffusion(K_ten, iterations=self.iterations, kNN=self.kNN)

    def predict_TKL(self, X_test_list):
        result = []
        self.K_list = []
        for i,X_test in enumerate(X_test_list):
            # X = np.vstack([self.X_train_list[i], X_test])
            index_train = np.array(range(self.X_train_list[i].shape[0]))
            index_test =  len(index_train) + np.array(range(X_test.shape[0]))
            K = np.copy(self.K_ten_dp[i,:,:])
            K = K/ np.sqrt(np.linalg.norm(K))
            
            K_train = K[np.ix_(index_train,index_train)]
            K_test = K[np.ix_(index_test,index_train)]
            
            clf = SVC(kernel='precomputed') 
            clf.fit(K_train, self.y_train)
            y_pred = clf.predict(K_test)
            result.append({f"modality": i, "y": y_pred})
        return result
    
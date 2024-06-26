# %%
import os  
import scipy.io
import numpy as np
import pandas as pd



from icecream import ic


from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,mean_squared_error,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

import scipy.io
import matplotlib.pyplot as plt

from tensorly.decomposition import parafac
import tensorly as tl


# from TKL import TKL
from KL import KL

from utils import *
# %%

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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from icecream import ic
import matplotlib.pyplot as plt



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
        K_ten_dp = cross_diffusion(K_ten, iterations=self.iterations, kNN=self.kNN)
        
        K_combine = np.sum(K_ten_dp,axis=0)
        # K_combine = np.squeeze(K_ten[1:,:])
        
        K_combine = K_combine/ np.sqrt(np.linalg.norm(K_combine))
        K_train = K_combine[np.ix_(index_train,index_train)]
        K_test = K_combine[np.ix_(index_test,index_train)]

        clf = SVC(kernel='precomputed') 
        clf.fit(K_train, self.y_train)
        y_pred = clf.predict(K_test)
        return y_pred,K_ten,K_ten_old,index_train,index_test



# load data
args_dataPathList = fr'D:\Code-git\DeepKernelLearning/data/AD_CN/GM.csv,D:\Code-git\DeepKernelLearning/data/AD_CN/PET.csv,D:\Code-git\DeepKernelLearning/data/AD_CN/CSF.csv,D:\Code-git\DeepKernelLearning/data/AD_CN/SNP.csv'
dataPathList = [str(item) for item in args_dataPathList.split(',')]
X_list = []
for dataPath in dataPathList:
  X = pd.read_csv(dataPath,header=None).values
  X_list.append(X)

df_y = pd.read_csv(fr'D:\Code-git\DeepKernelLearning/data/AD_CN/AD_CN_label.csv')    
y = df_y['encoded']
le = LabelEncoder()
y = le.fit_transform(y)

#split data   
exp = 5
Z = range(len(y)) 
Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2, random_state=exp)


# %% Single kernel

# for modality in range(4):
#   X_train = X_list[modality][Z_train,:]
#   X_test = X_list[modality][Z_test,:]

#   clf = KL()
#   clf.fit(X_train,y_train)

#   y_pred = clf.predict(X_test)
#   cf  = confusion_matrix(y_pred, y_test)
#   acc = round(np.sum(np.diag(cf))/ np.sum(cf)*100,2)
#   ic(f"Modality: {modality}",cf, acc)
# %%
X_train_list = [x[Z_train,:] for x in X_list]
X_test_list = [x[Z_test,:] for x in X_list] 
clf = TKL(iterations=3,n_tree=500, kNN=5)
clf.fit(X_train_list,y_train,val_size=0)

# %%
# y_pred = clf.predict(X_test_list)
# cf  = confusion_matrix(y_pred, y_test)
# acc = round(np.sum(np.diag(cf))/ np.sum(cf)*100,2)
# ic(acc, cf)

# %%
y_pred,K_ten,K_ten_old,index_train,index_test = clf.predict_(X_test_list)
# K_dp = cross_diffusion(K_ten,5,3)

# %%
cf  = confusion_matrix(y_pred, y_test)
acc = round(np.sum(np.diag(cf))/ np.sum(cf)*100,2)
ic("TKL prediction")
ic(acc, cf)

# %%
K = np.sum(K_ten,axis=0)
K_combine = K/ np.sqrt(np.linalg.norm(K))
K_train = K_combine[np.ix_(index_train,index_train)]
K_test = K_combine[np.ix_(index_test,index_train)]

clf = SVC(kernel='precomputed') 
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)
cf  = confusion_matrix(y_pred, y_test)
acc = round(np.sum(np.diag(cf))/ np.sum(cf)*100,2)
ic("Summation K ten")
ic(acc, cf)

# %%
K = np.sum(K_ten_old,axis=0)
K_combine = K/ np.sqrt(np.linalg.norm(K))
K_train = K_combine[np.ix_(index_train,index_train)]
K_test = K_combine[np.ix_(index_test,index_train)]

clf = SVC(kernel='precomputed') 
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)
cf  = confusion_matrix(y_pred, y_test)
acc = round(np.sum(np.diag(cf))/ np.sum(cf)*100,2)
ic("Summation K origin")

ic(acc, cf)
# %%
K_dp = cross_diffusion(K_ten)
K = np.sum(K_dp,axis=0)
K_combine = K/ np.sqrt(np.linalg.norm(K))
K_train = K_combine[np.ix_(index_train,index_train)]
K_test = K_combine[np.ix_(index_test,index_train)]

clf = SVC(kernel='precomputed') 
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)
cf  = confusion_matrix(y_pred, y_test)
acc = round(np.sum(np.diag(cf))/ np.sum(cf)*100,2)
ic("Summation K old + DP")

ic(acc, cf)
# %%
K_dp = cross_diffusion(K_ten,3,5)
K = np.sum(K_dp,axis=0)
K_combine = K/ np.sqrt(np.linalg.norm(K))
K_train = K_combine[np.ix_(index_train,index_train)]
K_test = K_combine[np.ix_(index_test,index_train)]

clf = SVC(kernel='precomputed') 
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)
cf  = confusion_matrix(y_pred, y_test)
acc = round(np.sum(np.diag(cf))/ np.sum(cf)*100,2)
ic("Summation K old + DP")

ic(acc, cf)
# %%
train_idx_sorted = np.argsort(y_train)
K_train_sorted = K_combine[np.ix_(index_train,index_train)][np.ix_(train_idx_sorted,train_idx_sorted)]
plt.imshow(K_train_sorted) 
# %%
y_combine = np.hstack((y_train, y_test))
idx_combine_sorted  = np.argsort(y_combine)


K = np.sum(K_ten,axis=0)
K_sorted = K[np.ix_(idx_combine_sorted,idx_combine_sorted)]
plt.imshow(K_sorted) 

# %%
y_combine = np.hstack((y_train, y_test))
idx_combine_sorted  = np.argsort(y_combine)

K_ten_sorted = np.zeros_like(K_ten_old)
for i in range(4):
  K_ten_sorted[i,:,:] = np.squeeze(K_ten[i,:,:])[np.ix_(idx_combine_sorted,idx_combine_sorted)]
K_dp = cross_diffusion(K_ten_sorted,3,5)

K_sorted = np.sum(K_dp,axis=0)

plt.imshow(K_sorted) 

new_index = np.array(range(len(y)))[idx_combine_sorted]
index_train_sorted = new_index[index_train]
index_test_sorted = new_index[index_test]
K_sorted = K_sorted/ np.sqrt(np.linalg.norm(K_sorted))
K_train = K_sorted[np.ix_(index_train_sorted,index_train_sorted)]
K_test = K_sorted[np.ix_(index_test_sorted,index_train_sorted)]

clf = SVC(kernel='precomputed') 
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)
cf  = confusion_matrix(y_pred, y_test)
acc = round(np.sum(np.diag(cf))/ np.sum(cf)*100,2)
ic("Summation K old + DP")


# %%
K_dp = cross_diffusion(K_ten,3,5)
y_combine = np.hstack((y_train, y_test))
idx_combine_sorted  = np.argsort(y_combine)


K = np.sum(K_dp,axis=0)
K_sorted = K[np.ix_(idx_combine_sorted,idx_combine_sorted)]
plt.imshow(K_sorted)
# %%

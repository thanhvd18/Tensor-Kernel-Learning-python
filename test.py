# %%
import os  

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


from TKL import TKL
from KL import KL

from utils import *
# load data
args_dataPathList = fr'D:Code-git\DeepKernelLearning/data/AD_CN/GM.csv,D:\Code-git\DeepKernelLearning/data/AD_CN/PET.csv,D:\Code-git\DeepKernelLearning/data/AD_CN/CSF.csv,D:\Code-git\DeepKernelLearning/data/AD_CN/SNP.csv'
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
Z = range(len(y)) 
Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2, random_state=0)


# %% Single kernel
modality = 0
X_train = X_list[modality][Z_train,:]
X_test = X_list[modality][Z_test,:]

clf = KL()
clf.fit(X_train,y_train)


# %%
X_train_list = [x[Z_train,:] for x in X_list]
X_test_list = [x[Z_test,:] for x in X_list] 
clf = TKL(iterations=5,n_tree=500, kNN=3)
clf.fit(X_train_list,y_train)

results  = clf.predict_individual_multimodal(X_test_list)
clf.TKL_train()

for d in results:
  accuracy = accuracy_score(y_test, d['y'])
  ic(accuracy)

results  = clf.predict_TKL(X_test_list)
for d in results:   
  accuracy = accuracy_score(y_test, d['y'])
  ic(accuracy)
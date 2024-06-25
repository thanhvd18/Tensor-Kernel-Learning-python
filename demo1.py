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
  
      
def classify_kernel(K,K_y,Z):
  train_index, val_index, test_index = Z[0,0],Z[0,1],Z[0,2] 
  train_index = np.hstack([train_index, val_index]).ravel().astype(int)
  test_index = test_index.ravel().astype(int)
  
  K = K/ np.sqrt(np.linalg.norm(K))
  K_train = K[np.ix_(train_index, train_index)]
  K_test = K[np.ix_(test_index,train_index)]
  y = K_y[0,:]
  y_train = y[train_index]
  y_test = y[test_index]


  clf = SVC(kernel='precomputed') 
  clf.fit(K_train, y_train)
  y_pred_late = clf.predict(K_test)
  accuracy_late = accuracy_score(y_test, y_pred_late)
  ic(accuracy_late)
  return accuracy_late


# %%

mat_file_path = fr"D:/Code-git/DeepKernelLearning/ADvsCN/K_X_1.mat"
mat_data = scipy.io.loadmat(mat_file_path)

K_X = np.copy(mat_data['K_X'])
K_y = np.squeeze(mat_data['K_y'])
Z = mat_data['Z']

K_X_copy = np.copy(K_X)
K_ten = parafac(K_X_copy, rank=2, tol=1e-9,n_iter_max=1000,init='random')
K_ten = tl.cp_to_tensor(K_ten)

K_ten = cross_diffusion(K_ten,3,5)
K = np.sum(K_ten, axis=0)
print("TKL + DP ***********")
classify_kernel(K,K_y,Z)
plt.imshow(K)


K = np.sum(mat_data['K_X'], axis=0)
print("Summation ***********")
classify_kernel(K,K_y,Z)
plt.figure()
plt.imshow(K)
# %%

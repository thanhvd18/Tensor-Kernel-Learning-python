import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from icecream import ic



def sparse_graph( W,kNN=5):
  W_i = np.copy(W)
  N = W_i.shape[0]
  S = np.zeros_like(W_i)
  for row in range(N):
    indices = np.argsort(-W_i[row, :])
    k_neighbors = indices[1:kNN+1]  # Skip the self-loop
    for j in k_neighbors:
      S[row, j] =   W_i[row, j] / np.sum(W_i[row, k_neighbors])
  S = S + S.T - np.diag(np.diag(S))
  return S
def similarity_normalization(W_i):
  # W = np.copy(W_i)
  W = W_i
  
  N = W.shape[0]
  W_i = 0.5 * np.eye(N)
  for row in range(N):
    row_sum = np.sum(W[row, :]) - W[row, row]
    for col in range(row + 1, N):
      W_i[row, col] = W[row, col] / (2 * row_sum)
  W_i = W_i + W_i.T - np.diag(np.diag(W_i))
  return W_i


def cross_diffusion(K_ten_, iterations=3,kNN=5):
  K_ten = np.copy(K_ten_)
  if K_ten.shape[0] == K_ten.shape[1]:
    m = K_ten.shape[2]
    S_ten = np.zeros_like(K_ten)
    K_norm_ten = K_ten
    for i in range(m):
      K_i = K_ten[:,:,i]
      K_norm_ten[:, :,i ] = similarity_normalization(K_i)
        
        
    
    for i in range(m):
      K_i = K_norm_ten[:, :, i]
      S_ten[:, :, i] = sparse_graph(K_i, kNN)



    for _ in range(iterations):
      for i in range(m):
        S_i = S_ten[:, :, i]
        exclude_i = [j for j in range(m) if j != i]
        sum_K = np.sum(K_norm_ten[:, :,exclude_i], axis=2)
        K_norm_ten[:, :, i] = S_i @ (1 / (m - 1) * sum_K) @ S_i.T
  elif K_ten.shape[-1] == K_ten.shape[1]:
    m = K_ten.shape[0]
    S_ten = np.zeros_like(K_ten)
    K_norm_ten = np.copy(K_ten)
    for i in range(m):
      K_i = np.copy(K_ten[i,:,:])
      K_norm_ten[i,:, : ] = np.copy(similarity_normalization(K_i))
        
        

    for i in range(m):
      K_i = np.copy(K_norm_ten[i,:, :])
      S_ten[i, :, :] = sparse_graph(K_i, kNN)



    for _ in range(iterations):
      for i in range(m):
        S_i = np.copy(S_ten[i,:, :])
        exclude_i = [j for j in range(m) if j != i]
        sum_K = np.sum(K_norm_ten[exclude_i,:, :], axis=0)
        K_norm_ten[i,:, :] = np.copy(S_i @ (1 / (m - 1) * sum_K) @ S_i.T)
  return K_norm_ten

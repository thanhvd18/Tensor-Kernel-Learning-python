import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


class KL:
  
  '''
    Kernel learning: Random Forest + Kernel SVM
  '''
  def __init__(self,n_trees=500):
    self.K = None
    self.Z = None
    self.n_trees = n_trees
    self.rf_classifier = None
    self.y_train = None
    self.X_train = None
  
  def train_test_split(self, test_size=0.2, random_state=0):
    pass
  def get_kernel_matrix(self, X, y=None, index=None):
    if self.rf_classifier is None and y is not None:
      self.fit(X,y)
    if y is not None:
      index = np.argsort(y)
      y = np.array(y)[index]
      X = np.array(X)[index,:]
    pairwise_similarities = [[0] * len(X) for _ in range(len(X))]

  
    # Iterate over each tree in the forest
    for tree in self.rf_classifier.estimators_:
      # Get the leaf indices for each example
      leaf_indices = tree.apply(X)

        # Update pairwise similarities based on leaf indices
      for i in range(len(X)):
          for j in range(i + 1, len(X)):
              if leaf_indices[i] == leaf_indices[j]:
                  pairwise_similarities[i][j] += 1
                  pairwise_similarities[j][i] += 1

    # Normalize the pairwise similarities by the number of trees
    normalized_pairwise_similarities = [[similarity / self.n_trees for similarity in row] for row in pairwise_similarities]
    K  = np.array(normalized_pairwise_similarities)
    return K


  def fit(self, X_train, y_train,random_state=0,indices=None):
    from sklearn.ensemble import RandomForestClassifier
    
      # Create a Random Forest classifier
    self.rf_classifier = RandomForestClassifier(n_estimators=self.n_trees, random_state=42)
                                        #  max_features=5,
                                        #  max_depth=5
    self.rf_classifier.fit(X_train,y_train)
    self.y_train = y_train
    self.X_train = X_train
                                      
  def predict(self, X_test):
      X = np.vstack([self.X_train, X_test])
      index_train = np.array(range(X_train.shape[0]))
      index_test =  len(index_train) + np.array(range(X_test.shape[0]))
      K = self.get_kernel_matrix(X)
      
      K_train = K[np.ix_(index_train,index_train)]
      K_test = K[np.ix_(index_test,index_train)]
      
      clf = SVC(kernel='precomputed') 
      clf.fit(K_train, self.y_train)
      y_pred = clf.predict(K_test)
      return y_pred
    
  def feature_selection(self):
    pass

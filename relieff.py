import numpy as np


def find_nearest_hit_miss(K, k_y):
    num_instances = K.shape[0]
    nearest_hit = []
    nearest_miss = []

    for i in range(num_instances):
        same_class_indices = np.where(k_y[0][i] == 1)[0]
        diff_class_indices = np.where(k_y[0][i] == -1)[0]

        # Exclude the instance itself
        same_class_indices = same_class_indices[same_class_indices != i]
        diff_class_indices = diff_class_indices[diff_class_indices != i]

        #Find nearest hit
        if len(same_class_indices) > 0:
            nearest_hit.append(same_class_indices[np.argmax(K[i, same_class_indices])])
        else:
            nearest_hit.append(-1)
        
        #Find nearest miss
        if len(diff_class_indices) > 0:
            nearest_miss.append(diff_class_indices[np.argmin(K[i, diff_class_indices])])
        else:
            nearest_miss.append(-1)

    return nearest_hit, nearest_miss

def relieff(X, y, nearest_hit, nearest_miss, num_features_select=20):
    num_instances = X.shape[0]
    num_features = X.shape[1]
    weights = np.zeros(num_features)
    for i in range(num_instances):
        instance = X[i,:]
        nearest_hit_instance = X[nearest_hit[i],:]
        nearest_miss_instance = X[nearest_miss[i],:]
        diff_hit = np.square(instance - nearest_hit_instance)
        diff_miss = np.square(instance - nearest_miss_instance)
        weights += diff_miss - diff_hit
    
    # ranking the features and select the top num_features
    feature_ranking = np.argsort(weights)
    return weights, feature_ranking[:num_features_select]

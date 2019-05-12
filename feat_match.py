'''
  File name: feat_match.py
  Author: Vaishak Kumar
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Output match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''
import numpy as np
import scipy.spatial 

def feat_match(descs1, descs2):
  threshold = 0.6 #Accept match if ratio of best match to second best is less than this threshold 
  match = np.empty(descs1.shape[1])
  
  a = scipy.spatial.KDTree(np.transpose(descs2))
  #For each interest point in the first image, find the two most similar neighbors
  #Test if the ratio is < threshold to see if it's a good match
  #If good match, put the index of the best match descriptor into match
  #If not a good, match put -1 to indicate no good match found
  #Closest neighbors is implemented using Annoy from Python to conduct nearest neighbor search
  for i in range(descs1.shape[1]):
    #Find the two closest neighbors
    distances,indices = a.query(descs1[:,i], k=2)
    if (distances[0] / distances[1]) < threshold:
      match[i] = int(indices[0])
    else:
      match[i] = int(-1)
  return match


'''
  File name: ransac_est_homography.py
  Author: Vaishak Kumar
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''
import numpy as np
from est_homography import est_homography

def ransac_est_homography(x1, y1, x2, y2, thresh):
  inlier_ind = np.zeros(len(x1))
  for i in range(1000):
    inlier_ind1 = np.zeros(len(x1))
    indices = []
    while len(set(indices))!=4:
        indices = np.random.randint(0,len(x1),4)
    x=np.array([x1[i] for i in indices])
    y=np.array([y1[i] for i in indices])
    X=np.array([x2[i] for i in indices])
    Y=np.array([y2[i] for i in indices])
    homography = est_homography(x, y, X, Y)
    for j in range(len(x1)):
        transform = np.matmul(homography,np.transpose([x1[j],y1[j],1]))
        if np.linalg.norm(transform*(1/transform[2])-np.transpose([x2[j],y2[j],1])) < thresh:
            inlier_ind1[j]=1
    if sum(inlier_ind1)>sum(inlier_ind):
        inlier_ind=inlier_ind1
  #Calculate final homography
  print(inlier_ind)
  x=np.array([x1[i] for i in range(len(x1)) if inlier_ind[i]])
  y=np.array([y1[i] for i in range(len(x1)) if inlier_ind[i]])
  X=np.array([x2[i] for i in range(len(x1)) if inlier_ind[i]])
  Y=np.array([y2[i] for i in range(len(x1)) if inlier_ind[i]])
  H = est_homography(x, y, X, Y)
  return H, inlier_ind

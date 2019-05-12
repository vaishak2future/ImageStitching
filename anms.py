'''
  File name: anms.py
  Author:Vaishak Kumar
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''
import numpy as np

def anms(cimg, max_pts):
  cimgFlat = []
  median = np.percentile(cimg,99.9)
  for i in range(cimg.shape[0]):
    for j in range(cimg.shape[1]):
      corner = cimg[i,j]
      if corner > median:
        cimgFlat.append((i,j,corner))
      
  dist = np.zeros(len(cimgFlat))
  dist = (dist + 1)*np.inf
  for i in range(len(cimgFlat)):
    for j in range(len(cimgFlat)):
      if cimgFlat[j][2] < 0.9*cimgFlat[i][2] or i==j:
        pass
      else:
        distance = np.linalg.norm(np.array([cimgFlat[i][0],cimgFlat[i][1]])-np.array([cimgFlat[j][0],cimgFlat[j][1]]))
        if distance<dist[i]:
            dist[i]=distance
  sorted_list_indices=np.argsort(dist)[-max_pts:]
  x=[]
  y=[]
  rmax=dist[sorted_list_indices[-1]]
  for i in sorted_list_indices:
      x.append(cimgFlat[i][1])
      y.append(cimgFlat[i][0])
  return x, y, rmax

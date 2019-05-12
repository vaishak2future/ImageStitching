'''
  File name: feat_desc.py
  Author: Vaishak Kumar
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''
import numpy as np
import scipy.ndimage.filters
def feat_desc(img, x, y):
  # Your Code Here
  descs=np.zeros((64,len(x)))
  for a in range(len(x)):
    desc=[]
    x1=x[a]
    y1=y[a]
    patch = np.empty((40,40))
	#create 40x40 patch
    patchx=0
    patchy=0
    for j in range(x1-20,x1+20):
        for i in range(y1-20,y1+20):
            #reflect if i or j exceed bounds
            i_new = np.abs(i)+min(0,2*(img.shape[0]-1-np.abs(i)))
            j_new = np.abs(j)+min(0,2*(img.shape[1]-1-np.abs(j)))
            patch[patchy,patchx]=img[i_new,j_new]
            patchy=patchy+1
        patchy=0
        patchx=patchx+1
    #Blur the patch
    patch = scipy.ndimage.filters.gaussian_filter(patch,2)
    #Loop through every 5th pixel in 40X40 patch
    for j in range(0,patch.shape[1],5):
        for i in range(0,patch.shape[0],5):
            desc.append(patch[i,j])
	#normalize vector
    desc=desc-np.mean(desc)
    desc=desc/np.sqrt((np.sum(desc**2)))
	#add to array
    descs[:,a]=desc
  return descs

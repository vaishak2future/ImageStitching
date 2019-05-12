'''
Points Plotting for Detection:
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from corner_detector import corner_detector
from anms import anms
from ransac_est_homography import ransac_est_homography 
from feat_desc import feat_desc
from feat_match import feat_match
'''
Corner Harris Image Display 
'''
im1 = cv2.imread("1L.png")
img = np.float32(cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY))
cimg = corner_detector(img)
cimgRGB = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
'''
implot = plt.imshow(cimgRGB)
x = []
y = []
for i in range(cimg.shape[0]):
  for j in range(cimg.shape[1]):
    if cimg[i,j] >= np.percentile(cimg,99):
      y.append(i)
      x.append(j)
plt.scatter(x, y, c='r', s=40)
plt.show()
#cv2.imwrite("cornerHarris.png",cimg)
'''

'''
Adaptive NMS Image Display
'''
'''
x, y, rmax = anms(cimg,1000)
implot = plt.imshow(cimgRGB)
plt.scatter(x, y, c='r', s=40)
plt.show()
'''


'''
post-RANSAC matching (outliers in blue dots) for at least five distinct frames.
'''
img1 = cv2.imread("1L.png")
img2 = cv2.imread("1M.png")
im1 = np.float32(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY))
im2 = np.float32(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY))

img1RGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#Detecting Corners
corners1=corner_detector(im1)
corners2=corner_detector(im2)

x1,y1,z1=anms(corners1, 1000)
x2,y2,z2=anms(corners2, 1000)


descs1 =feat_desc(im1, x1, y1)
descs2 =feat_desc(im2, x2, y2)
    
matches = feat_match(descs1,descs2)
matched_x2 = []
matched_y2 = []
matched_x1 = []
matched_y1 = []
for i in range(len(matches)):
    if matches[i]!=-1:
        matched_x1.append(x1[i])
        matched_y1.append(y1[i])
        matched_x2.append(x2[int(matches[i])])
        matched_y2.append(y2[int(matches[i])])
        
H,inliers = ransac_est_homography(matched_x1,matched_y1,matched_x2,matched_y2,5) 

implot = plt.imshow(img1RGB)

bluex1 = []
bluey1 = []
redx1 = []
redy1 = []

bluex2 = []
bluey2= []
redx2 = []
redy2 = []
for i in range(len(matched_x1)):
  if inliers[i] == 0:
    bluex1.append(matched_x1[i])
    bluey1.append(matched_y1[i])
    
    bluex2.append(matched_x2[i])
    bluey2.append(matched_y2[i])
  else:
    redx1.append(matched_x1[i])
    redy1.append(matched_y1[i])
    
    redx2.append(matched_x2[i])
    redy2.append(matched_y2[i])
  
plt.scatter(bluex1, bluey1, c='b', s=40)
plt.scatter(redx1, redy1, c='r', s=40)
plt.show()
implot = plt.imshow(img2RGB)
plt.scatter(bluex2, bluey2, c='b', s=40)
plt.scatter(redx2, redy2, c='r', s=40)
plt.show()


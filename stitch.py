import cv2
import matplotlib.pyplot as plt
import numpy as np
from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from PIL import Image

def stitch(im1,im2):
    gray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray1 = np.float32(gray1)
    gray2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    gray2 = np.float32(gray2)
    
    
    print("corners")
    corners1=corner_detector(gray1)
    corners2=corner_detector(gray2)
    
    print("anms")
    x1,y1,_=anms(corners1, 1000)
    x2,y2,_=anms(corners2, 1000)
    
    
    print("feat_desc")
    descs1 =feat_desc(gray1, x1, y1)
    descs2 =feat_desc(gray2, x2, y2)
    
    
    print("match")
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
            
    print("RANSAC")
    H,inliers = ransac_est_homography(matched_x1,matched_y1,matched_x2,matched_y2,5)

    #Find outer bounds and displacements
    x,y=[],[]
    corners = [(0,0),(0,im1.shape[1]-1),(im1.shape[0]-1,0),(im1.shape[0]-1,im1.shape[1]-1)]
    for i in corners:
        transform = np.matmul(H,np.transpose([i[1],i[0],1]))
        transform = transform*(1/transform[2])
        x.append(transform[0])
        y.append(transform[1])
    if np.array(y).min()<0:
        dispy= np.array(y).min()
    else:
        dispy=0
    if np.array(x).min()<0:
        dispx= np.array(x).min() 
    else:
        dispx=0
    warped_img = np.zeros((int(np.ceil(max(gray2.shape[0],np.array(y).max())+np.abs(dispy))),int(np.ceil(max(gray2.shape[1],np.array(x).max())+np.abs(dispx))),3))
    inverse = np.linalg.inv(H)
    print("stitching")
    for i in range(warped_img.shape[0]):
        for j in range (warped_img.shape[1]):
            transform = np.matmul(inverse,np.transpose([j+dispx,i+dispy,1]))
            transform = transform*(1/transform[2])
            if i in range(gray2.shape[0]) and j in range(gray2.shape[1]):      
                for channel in range(3):
                    warped_img[i+int(np.abs(np.ceil(dispy))),j+int(np.abs(np.ceil(dispx))),channel]=im2[i,j,channel]        
            if int(transform[0]) in range(gray1.shape[1]) and int(transform[1]) in range(gray1.shape[0]):      
                i_wt=np.ceil(transform[1])-transform[1]
                j_wt=np.ceil(transform[0])-transform[0]
                try:
                    for channel in range(3):
                        warped_img[i,j,channel]=(i_wt*j_wt)*im1[int(transform[1]),int(transform[0]),channel]+(1-i_wt)*j_wt*im1[int(transform[1])+1,int(transform[0]),channel]+(1-j_wt)*i_wt*im1[int(transform[1]),int(transform[0])+1,channel]+(1-j_wt)*(1-i_wt)*im1[int(transform[1])+1,int(transform[0])+1,channel]
                except:
                    for channel in range(3):
                        warped_img[i,j,channel]=im1[int((transform[1])),int((transform[0])),channel]
    cv2.imwrite("intermediate.jpg",warped_img)
    warped_img = cv2.imread("intermediate.jpg")
    return warped_img




'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''
from stitch import stitch
import cv2

def mymosaic(img_input):
  if len(img_input)==1:
      return img_input[0]
  next=[]
  for i in range(len(img_input)-1):
      if i%2==0:
          next.append(stitch(img_input[i],img_input[i+1]))
      else:
          next.append(stitch(img_input[i+1],img_input[i]))
  return mymosaic(next)

im1 = cv2.imread("1.jpg")
im2 = cv2.imread("2.jpg")
im3 = cv2.imread("3.jpg")
cv2.imwrite("result12.jpg",mymosaic([im1,im2,im3]))

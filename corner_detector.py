'''
  File name: corner_detector.py
  Author:Vaishak Kumar
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

def corner_detector(img):
  cimg = cv2.cornerHarris(img,2,3,0.2)
  return cimg



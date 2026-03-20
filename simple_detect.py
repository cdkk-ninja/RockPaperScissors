import cv2
import numpy as np
import math
import logging

from sys import path
path.append('../cckk')
import cckkCV

config = {
    "image_path": "images/",
}

cckkCV.cckkORB.read_colour()

img = cckkCV.cckkORB(img_filename = "RockBW.jpg", img_path = config['image_path'], auto_detect=False)
print(f"Identified shape: {img.identify_shape()}")

img = cckkCV.cckkORB(img_filename = "PaperBW.jpg", img_path = config['image_path'], auto_detect=False)
print(f"Identified shape: {img.identify_shape()}")

img = cckkCV.cckkORB(img_filename = "ScissorsBW.jpg", img_path = config['image_path'], auto_detect=False)
print(f"Identified shape: {img.identify_shape()}")

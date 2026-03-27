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

cckkCV.cckkCV2Detect.read_colour()

file_list = ["RockBW.jpg", "PaperBW.jpg", "ScissorsBW.jpg", "RockW.jpg", "PaperW.jpg", "ScissorsW.jpg", "Rock.jpg", "Paper.jpg", "Scissors.jpg"]

for filename in file_list:
    img = cckkCV.cckkCV2Detect(img_filename = filename, img_path = config['image_path'])
    print(f"Identified shape: {img.identify_shape()} for {img.filename}")

for filename in file_list:
    img = cckkCV.cckkCV2Detect(img_filename = filename, img_path = config['image_path'])
    print(f"Identified colour: {img.identify_colour()} for {img.filename}")

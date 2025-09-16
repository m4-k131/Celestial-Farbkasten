import os 
import cv2 
import numpy as np 

TEMPLATE = [
    {}
]

SINGLE_IMAGE = {
    "path" : {
        "color": [],
        "factor": 1,
        "operation": "add"
    }
}
TRANSFORMATION = np.zeros(3,3,3)

def combine(images_dict,  outname):
    image = np.zeros((3,3,3))

    pass
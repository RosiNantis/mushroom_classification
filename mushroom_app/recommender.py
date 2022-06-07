"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

from email.mime import image
import os
import pandas as pd
import numpy as np
import random
import cv2
from utils import image_classification

def mushroom_classification(image_dir ):
    """
    return k first probabl mushroom identifications
    """
    direct = '../data/test/'
    image_dir = direct + str(image_dir)
    dataFram = image_classification(image_dir)
    mushrooms_pred = dataFram.values.tolist()[:5]
    return mushrooms_pred, dataFram.image_class, dataFram['probability(%)']

def mushroom_prediction(image_dir):
    """
    returns the prediction of the mushroom
    """
    direct = '../data/test/'
    image_dir = direct + str(image_dir)
    dataFram = image_classification(image_dir)
    return dataFram.image_class[0]

def mushroom_depict(image_dir):
    """
    returns the image of test mushroom
    """
    direct = '../data/test/'
    image_dir = direct + str(image_dir)
    item = cv2.imread(image_dir)
    #plt.imshow(item,cmap='Greys')
    return image_dir

#mushrooms_pred, k, j = mushroom_classification('003_eIMrvDdKleY.jpg')

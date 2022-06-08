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
    mushrooms_pred = dataFram.values.tolist()[:4]
    mushrooms_pred_list = pd.DataFrame(dataFram)
    return mushrooms_pred, dataFram.image_class, dataFram['probability(%)'], mushrooms_pred_list[:4]

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

# mushrooms_pred, k, j, df = mushroom_classification('003_eIMrvDdKleY.jpg')
# print(df)
# print(df['image_class'])
# print(df['probability(%)'])
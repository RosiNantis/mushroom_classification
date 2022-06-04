# import streamlit as st

# header - st.beta_container()
# dataset - st.beta_container()

# with header :
#     st.title('Welcome to the Data Explorer App')

#import std libraries
from cv2 import FileStorage_INSIDE_MAP
import pandas as pd
import numpy as np
from soupsieve import select
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Write a title
st.title('Welcome to the Mushroom Classifier Application ')
# Write data taken from https://allisonhorst.github.io/palmerpenguins/
st.write('Nature alone is antique, and the oldest art a mushroom')
st.write('>Thomas Carlyle<')
st.write('The data were collected from **Kaggle**: [:mushroom:](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) & [:mushroom:](https://www.kaggle.com/competitions/fungiclef2022/data) ')
# Put image https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/man/figures/lter_penguins.png
st.image('mushroom_forest.jpg')
# Write heading for Data
st.header('Data')

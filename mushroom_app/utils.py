"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""

# all imports 
import numpy as np
from os import listdir, walk
from os.path import isfile, join
import pandas as pd
from tensorflow import keras
#from transformers import PegasusForConditionalGeneration


CLASSES = {'Mycena_galericulata': 0,
 'Trametes_versicolor': 1,
 'Plicatura_crispa': 2,
 'Tremella_mesenterica': 3,
 'Hypholoma_fasciculare': 4,
 'Pluteus_cervinus': 5,
 'Meripilus_giganteus': 6,
 'Pleurotus_ostreatus': 7,
 'Russula': 8,
 'Stereum_hirsutum': 9}

def evaluate(model_name):
    model=keras.models.load_model("../models/" + model_name + ".h5")
    history = pd.read_csv('../models/history_' + model_name + '.csv')
    # plot_history(history)
    acc = history.tail(1)
    print(f"The model has an accuracy of {int(round(acc['val_categorical_accuracy'].values[0],2)*100)} %.")
    return history, acc, model

def image_classification(image_path):
    image = keras.preprocessing.image.load_img(image_path, target_size=(224,224))
    pic_array = keras.preprocessing.image.img_to_array(image)
    image_batch = np.expand_dims(pic_array, axis=0)
    processed_image = keras.applications.mobilenet_v2.preprocess_input(image_batch)
    model_ann=keras.models.load_model("../models/" + 'MobileNetV2_3la_0001r_60bs_10set_f' + ".h5")
    probs = model_ann.predict(processed_image)[0].tolist()
    zipped = sorted(list(zip(CLASSES, probs)), key=lambda x: x[1], reverse=True)
    image_class = [zipped[i][0] for i in range(len(zipped))]
    probability  = [zipped[i][1]*100 for i in range(len(zipped))]
    probability = np.round(probability,0)
    df = pd.DataFrame(data={'image_class':image_class, 'probability(%)': probability})
    print(df)
    return df

# def initiate_models():
#     model_ann=keras.models.load_model("../models/" + 'ResNet50V2_2la_0001r_60bs_10set' + ".h5")
#     model_pegasus = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
#     return model_ann, model_pegasus

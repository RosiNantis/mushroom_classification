# mushroom_recommendation
An image recognition project where different Deep Learning models are deployed to be able to recognise the species of Fungis given in picture or from camera.

This project was created as a final project for graduating from the SPICED ACADEMY, Data Science. In this app I used data from two different Kaggle projects where I used 10 different Fungi species of 1000 images from each one. The data were resized in 224*224 in RGB and then split in (75/15/10) % train/validation/test data respectively. I used several pre-trained CNN models and after tuning of the hyper parameters I concluded that the best model is the MobileNetV2. As a next step I trained the model with 3 extra layers resulting in an accuracy of 76%. I used Flask to create an app that can identify Fungis either from the list of test images or from a web-camera. The prediction finally was linked with the respective Wikipedia page where the description of the Fungi is given along with a picture of it, for comparison. Furthermore I used NLP in order to make a short abstract of the description of the Fungi.

<img src="mushroom_app_gif.gif" width="1000"/>




### Licence

(c) 2022 Alexandros Samartzis

Distributed under the conditions of the MIT License. See License for details.
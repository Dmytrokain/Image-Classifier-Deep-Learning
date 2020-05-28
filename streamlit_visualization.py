import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def model_from_json():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model/model.h5")

    loaded_model.compile(optimizer='adam',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
    return loaded_model


def plot_image(i, predictions_array, true_label, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[0], cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("Predicted:{} {:2.0f}% (Real:{})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
  st.pyplot()


model = model_from_json()

st.text('Test Classifier')
imageIndex = st.slider('Choose image from test set', 0, 9999)


if st.button('Make prediction'):
  img = test_images[imageIndex]
  label = test_labels[imageIndex]
  img = (np.expand_dims(img,0))
  predictions_single = model.predict(img)
  plot_image(imageIndex, predictions_single, label, img)

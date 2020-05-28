import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json
from matplotlib import pyplot as plt


def save_model(trained_model):
    with open("model/model.json", "w") as json_file:
        json_file.write(trained_model.to_json())

    trained_model.save_weights("model/model.h5")


def save_loss_data(history):
    def convert_to_float(arr):
        for i in range(len(arr)):
            arr[i] = np.float64(arr[i]).item()
        return arr

    loss = convert_to_float(history.history['loss'])
    accuracy = convert_to_float(history.history['accuracy'])

    with open("model/loss_function.json", "w") as file:
        json.dump({'loss': loss, 'accuracy': accuracy}, file)


def loss_data_from_json():
    with open("model/loss_function.json", "r") as file:
        loss_dict = json.load(file)

    return loss_dict


def plot_loss_function(loss, accuracy):
    plt.plot(loss)
    plt.plot(accuracy)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


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


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def saved_model_present():
    return ("model.json" and "model.h5") in os.listdir("model/")


def test(model, test_images, test_labels):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    results = {'T-shirt/top': [0, 0], 'Trouser': [0, 0], 'Pullover': [0, 0], 'Dress': [0, 0], 'Coat': [0, 0],
               'Sandal': [0, 0], 'Shirt': [0, 0], 'Sneaker': [0, 0], 'Bag': [0, 0], 'Ankle boot': [0, 0]}

    for i in range(0, 9900, 10):
        img = test_images[i]
        lable = test_labels[i]
        img = (np.expand_dims(img, 0))
        predictions_single = model.predict(img)

        predictedLabel = np.argmax(predictions_single[0])

        data = results[class_names[lable]]
        if predictedLabel == lable:
            data[1] += 1
        else:
            data[0] += 1
        results[lable] = data
        # print(i)
    for key in results:
        data = results[key]
        persents = (data[1] * 100) / (data[0] + data[1])
        print(key, ": ", persents, "% success")


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0

if saved_model_present():
    model = model_from_json()
    loss_data = loss_data_from_json()
    loss, accuracy = loss_data['loss'], loss_data['accuracy']

    test(model, test_images, test_labels)
    plot_loss_function(loss, accuracy)
else:
    model = create_model()
    history = model.fit(train_images, train_labels, epochs=10)
    save_model(model)
    save_loss_data(history)

    plot_loss_function(history.history['loss'], history.history['accuracy'])

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
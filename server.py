from flask import Flask, request
from flask import render_template
import numpy as np
import pickle
import os

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
# import numpy as np
from PIL import Image
from skimage import io,transform


app = Flask(__name__)


def load_transform_data():
    (x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
    total_x0 = np.concatenate((x_train_org,x_test_org))
    total_y = np.concatenate((y_train_org,y_test_org))
    total_x = [0]*len(total_x0)
    for i in range(len(total_x0)):
      total_x[i] = transform.resize(total_x0[i],(img_rows,img_cols,1))
    total_x = np.asarray(total_x)

    return total_x, total_y

def split_data(num_classes, total_x, total_y):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(num_classes):
      index = np.where(total_y == i)
      train_index = np.random.choice(index[0], len(index[0])*4//5, replace=False)
      test_index = [i for i in index[0] if i not in train_index]

      x_train.extend(total_x[train_index])
      y_train.extend(total_y[train_index])
      x_test.extend(total_x[test_index])
      y_test.extend(total_y[test_index])

    total_x,total_y = [] , []
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test

def normalize_reshape_data(x_train, x_test):
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, x_test

def onehotencoding(y_train,y_test,num_classes):
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return y_train, y_test

def define_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def model_compile(model):

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    # return model

def fit_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, reduce_lr):
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[reduce_lr])

    return history

def compare_accuracies(history):

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()

def compare_loss(history):

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()


def save_model(model, path):
    # if os.path.exists(path):
    #     model.save(path+'/my_model')
    # write_file = open(path, 'wb')
    # model.save(path+'/my_model')
    model.save_weights(path)

def load_model(loaded_model,path):


    loaded_model.load_weights('saved_weights.h5')
    # model = tf.keras.models.load_model(path)




def crop_image(arr):
    arr = np.reshape(arr, (arr.shape[0],arr.shape[1],1))
    vert = arr[1:,:,:] - arr[:-1,:,:]
    horz = arr[:,1:,:] - arr[:,:-1,:]
    vcheck = np.sum(np.sum(vert,axis = 1),axis = 1)
    vind = np.nonzero(vcheck)
    vert_len = vind[0][-1] - vind[0][0]
    vlimit = vert_len // 5

    vstart = max(0,vind[0][0]-vlimit)
    vend = min(arr.shape[0]-1,vind[0][-1] + vlimit)
    hcheck = np.sum(np.sum(horz,axis = 0),axis = 1)
    hind = np.nonzero(hcheck)
    horz_len = hind[0][-1] - hind[0][0]
    hlimit = horz_len // 2

    hstart = max(0,hind[0][0]-hlimit)
    hend = min(arr.shape[1]-1,hind[0][-1] + hlimit)
    sample = arr[vstart:vend,hstart:hend,:]

    return sample

img_rows, img_cols = 75, 75
num_classes = 10
input_shape = (img_rows, img_cols, 1)

#### Unblock the below commented code to train on mnist dataset and save the model ####
#
# total_x,total_y = load_transform_data()
# x_train, y_train, x_test, y_test = split_data(num_classes, total_x, total_y)
# x_train, x_test = normalize_reshape_data(x_train, x_test)
# y_train, y_test = onehotencoding(y_train,y_test,num_classes)
#
# model = define_model()
#
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=5, min_lr=0.000001, verbose=1)
# batch_size = 128
# epochs = 32
# history = fit_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, reduce_lr)
#
# ## plotting accuracies and losses of training and validation datasets
# compare_accuracies(history)
# compare_loss(history)
#
# save the model
# save_path = 'saved_weights.h5'

# ## save the model
## save_path = 'saved_model75'
# save_model(model, save_path)
#
#########################################################################################


## loading the model

# loading_path = 'saved_weights.h5'
# loaded_model = load_model(loading_path)
# loaded_model = load_model('saved_model75')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


loaded_model = define_model()
# load_model(loaded_model,path)
loaded_model.load_weights('saved_weights.h5')

# print(loaded_model.summary())


import base64
from io import BytesIO
# import scipy.ndimage as sd

def predict(image):

    arr = image[:,:,0]

    cropped_image = crop_image(arr)

    image = cv2.resize(cropped_image,(img_rows,img_cols))

    image = image/255

    image = np.expand_dims(image,axis = 2)
    image = np.expand_dims(image,axis = 0)
    predicted = np.argmax(loaded_model.predict(image))
    return predicted

# image = 0

@app.route("/")
def init():
    return render_template('index.html')



@app.route("/process",methods = ['POST', 'GET'])
def process():
    global image
    global img_rows
    global img_cols
    global loaded_model

    imageData = request.form.get('image')
    imageData = imageData.replace('data:image/png;base64,','')
    imageData = imageData.replace(' ','+')
    imageBytes = base64.b64decode(imageData)
    img = Image.open(BytesIO(imageBytes))                           #.convert('RGB')

    img  = np.asarray(img)

    predicted = predict(img)

    return str(predicted)

if __name__ == "__main__":
    app.run()

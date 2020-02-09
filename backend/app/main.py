from flask import Flask
from flask import request
import base64
import sys
import tensorflow as tf
import numpy as np
import io
import json
import cv2 as cv2
from PIL import Image
from models.unet import get_unet
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from keras.optimizers import Adam

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import backend as K

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2 as cv2




app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World from Flask in a uWSGI Nginx Docker container with \
     Python 3.7 (from the example template)"


@app.route("/tfversion")
def tfversion():
    return tf.__version__

@app.route("/watermask", methods=['POST'])
def watermask():
    """

    """
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    # get images in base64 combert to numpy image with opencv
    # resize image to expected model call predict
    jsonData = json.loads((request.data.decode()))
    image = jsonData['image'].split(',')[1]
    base = base64.b64decode(image)
    
    iob = io.BytesIO(base)
    iob.seek(0)
    image_b = Image.open(iob)
    img = np.array(image_b)
    # b64Image = request.data['inputimage']
    print(img.shape)
    initial_h = img.shape[0]
    initial_w = img.shape[1]
    x_img = img_to_array(img)
    expect_h = 128
    expect_w = 128
    x_img = resize(x_img, (expect_h, expect_w, 3), mode='constant', preserve_range=True) / 255
    input_img = Input((expect_h, expect_w, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

    # model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    # Load best model
    model.load_weights('model-sea.h5')
    pred = np.zeros((1, expect_h, expect_w, 3))
    pred[0] = x_img

    preds_val = model.predict(pred, verbose=1)

    preds_val_t = (preds_val > 0.5).astype(np.uint8)

    result_img = resize(preds_val_t[0], (initial_h, initial_w, 3), mode='constant', preserve_range=True)

    rawBytes = io.BytesIO()
    im = Image.fromarray((result_img*255).astype("uint8"))
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)
    return base64.b64encode(rawBytes.read())


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=5000, threaded=False)

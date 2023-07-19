import warnings

warnings.simplefilter("ignore")

import imp
import numpy as np
import matplotlib.pyplot as plot
import os

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

tf.compat.v1.disable_eager_execution()

import innvestigate
import innvestigate.layers as ilayers
import innvestigate.backend as ibackend
import innvestigate.backend.checks as kchecks
import innvestigate.backend.graph as kgraph

# Use utility libraries to focus on relevant iNNvestigate routines.
import utils as eutils
import utils.imagenet as imagenetutils

# Build the model.
model = VGG16()
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Handle input depending on model.
channels_first = kbackend.image_data_format() == "channels_first"
color_conversion = "BGRtoRGB"  # keras.applications use BGR format

# Create model without trailing softmax
model_wo_sm = innvestigate.model_wo_softmax(model)

# Get some example test set images.
image_shape = model.input_shape[1:]
images, label_to_class_name = eutils.get_imagenet_data(image_shape[0])

if not len(images):
    raise Exception(
        "Please download the example images using: "
        "'innvestigate/examples/images/wget_imagenet_2011_samples.sh'"
    )

# Choose a sample image and add batch axis
image = images[0][0][None, :, :, :]
# Preprocess it for VGG16
image_pp = preprocess_input(image)
# Displaying the image
plot.imshow(image[0] / 255)
plot.show()
















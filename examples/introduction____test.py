import warnings

import imp
import os

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
# from tf.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import src.innvestigate as innvestigate
# Use utility libraries to focus on relevant iNNvestigate routines.
import utils.mnist as mnistutils

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

tf.compat.v1.disable_eager_execution()

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Create preprocessing functions
input_range = [-1, 1]
preprocess, revert_preprocessing = mnistutils.create_preprocessing_f(
    x_train, input_range
)

# Preprocess data
data = (
    preprocess(x_train),
    y_train,
    preprocess(x_test),
    y_test,
)

num_classes = len(np.unique(data[1]))
label_to_class_name = [str(i) for i in range(num_classes)]


# Create & train model
input_shape = (28, 28, 1)

model = keras.models.Sequential(
    [
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

scores = mnistutils.train_model(model, data, batch_size=128, epochs=2)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))

# Choosing a test image for the tutorial:
image = data[2][7:8]
plot.imshow(image.squeeze(), cmap="gray", interpolation="nearest")
plot.show()

# Stripping the softmax activation from the model
model_wo_sm = innvestigate.model_wo_softmax(model)
# # Creating an analyzer
# gradient_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)
# # Applying the analyzer
# analysis = gradient_analyzer.analyze(image)
# # Displaying the gradient
# plot.imshow(analysis.squeeze(), cmap="seismic", interpolation="nearest")
# plot.show()
#
# # Creating an analyzer
# gradient_analyzer = innvestigate.create_analyzer("gradient", model_wo_sm)
# # Applying the analyzer
# analysis = gradient_analyzer.analyze(image)
# # Displaying the gradient
# plot.imshow(analysis.squeeze(), cmap="seismic", interpolation="nearest")
# plot.show()

# Creating a parameterized analyzer
abs_gradient_analyzer = innvestigate.create_analyzer(
    "gradient", model_wo_sm, postprocess="abs"
)
square_gradient_analyzer = innvestigate.create_analyzer(
    "gradient", model_wo_sm, postprocess="square"
)
# Applying the analyzers
abs_analysis = abs_gradient_analyzer.analyze(image)
square_analysis = square_gradient_analyzer.analyze(image)
# Displaying the analyses, use gray map as there no negative values anymore
plot.imshow(abs_analysis.squeeze(), cmap="gray", interpolation="nearest")
plot.show()
plot.imshow(square_analysis.squeeze(), cmap="gray", interpolation="nearest")
plot.show()

# Creating an analyzer and set neuron_selection_mode to "index"
inputXgradient_analyzer = innvestigate.create_analyzer(
    "input_t_gradient", model_wo_sm, neuron_selection_mode="index"
)
for neuron_index in range(10):
    print("Analysis w.r.t. to neuron", neuron_index)
    # Applying the analyzer and pass that we want
    analysis = inputXgradient_analyzer.analyze(image, neuron_index)

    # Displaying the gradient
    plot.imshow(analysis.squeeze(), cmap="seismic", interpolation="nearest")
    plot.show()


import warnings

import imp
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plot
import tensorflow.keras.backend
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

tf.compat.v1.disable_eager_execution()

import innvestigate
import innvestigate.utils as iutils

# Use utility libraries to focus on relevant iNNvestigate routines.
import utils as eutils
import utils.imagenet as imagenetutils

# Load the model definition.
model = VGG16(weights="imagenet")
model.summary()

# Handle input depending on model and backend.
channels_first = keras.backend.image_data_format() == "channels_first"
color_conversion = "BGRtoRGB"  # keras.applications use BGR format

# Get some example test set images.
image_shape = [224, 224]
images, label_to_class_name = eutils.get_imagenet_data(size=image_shape[0])

if not len(images):
    raise Exception(
        "Please download the example images using: "
        "'innvestigate/examples/images/wget_imagenet_2011_samples.sh'"
    )

input_range = (-128, 128)  # format used by keras.applications
noise_scale = (input_range[1] - input_range[0]) * 0.1

# Methods we use and some properties.
# fmt: off
methods = [
    # NAME                  OPT.PARAMS                  POSTPROC FXN            TITLE
    # Show input.
    ("input",               {},                         imagenetutils.image,    "Input"),
    # Function
    ("gradient",            {"postprocess": "abs"},     imagenetutils.graymap,  "Gradient"),
    ("smoothgrad",          {"augment_by_n": 64,
                             "noise_scale": noise_scale,
                             "postprocess": "square"},  imagenetutils.graymap,  "SmoothGrad"),
    # Signal
    ("deconvnet",           {},                         imagenetutils.bk_proj,  "Deconvnet"),
    ("guided_backprop",     {},                         imagenetutils.bk_proj,  "Guided Backprop"),
    # Interaction
    ("deep_taylor.bounded", {"low": input_range[0],
                             "high": input_range[1]},   imagenetutils.heatmap,  "DeepTaylor"),
    ("input_t_gradient",    {},                         imagenetutils.heatmap,  "Input * Gradient"),
    ("integrated_gradients",
                            {"reference_inputs": input_range[0],
                             "steps": 64},              imagenetutils.heatmap,  "Integrated Gradients"),
    ("lrp.z",               {},                         imagenetutils.heatmap,  "LRP-Z"),
    ("lrp.epsilon",         {"epsilon": 1},             imagenetutils.heatmap,  "LRP-Epsilon"),
    ("lrp.sequential_preset_a_flat",
                            {"epsilon": 1},             imagenetutils.heatmap,  "LRP-PresetAFlat"),
    ("lrp.sequential_preset_b_flat",
                            {"epsilon": 1},             imagenetutils.heatmap,  "LRP-PresetBFlat"),
]
# fmt: on

# Create model without trailing softmax
model_wo_softmax = innvestigate.model_wo_softmax(model)

analyzers = []
for method in methods:
    a = method[0]
    try:
        analyzer = innvestigate.create_analyzer(
            method[0],  # analysis method identifier
            model_wo_softmax,  # model without softmax output
            **method[1]
        )  # optional analysis parameters
    except innvestigate.NotAnalyzeableModelException:
        # Not all methods work with all models.
        analyzer = None
    analyzers.append(analyzer)

# analyzer = innvestigate.create_analyzer(
#             "lrp.sequential_preset_b_flat",
#             model_wo_softmax,
#             epsilon=1
#         )


analysis = np.zeros([len(images), len(analyzers)] + image_shape + [3])
text = []


for i, (x, y) in enumerate(images):
    # Add batch axis.
    x = x[None, :, :, :]
    # x_pp = preprocess_input(x)

    # # Predict final activations, probabilites, and label.
    # presm = model_wo_softmax.predict_on_batch(x_pp)[0]
    # prob = model.predict_on_batch(x_pp)[0]
    # y_hat = prob.argmax()

    # # Save prediction info:
    # text.append(
    #     (
    #         "%s" % label_to_class_name[y],  # ground truth label
    #         "%.2f" % presm.max(),  # pre-softmax logits
    #         "%.2f" % prob.max(),  # probabilistic softmax output
    #         "%s" % label_to_class_name[y_hat],  # predicted label
    #     )
    # )

    for aidx, analyzer in enumerate(analyzers):
        if methods[aidx][0] == "input":
            # Do not analyze, but keep not preprocessed input.
            a = x / 255
        elif analyzer:
            # Analyze.
            a = analyzer.analyze(x)

            # Apply common postprocessing, e.g., re-ordering the channels for plotting.
            a = imagenetutils.postprocess(a, color_conversion, channels_first)
            # Apply analysis postprocessing, e.g., creating a heatmap.
            a = methods[aidx][2](a)
        else:
            a = np.zeros_like(images[i])
        # Store the analysis.
        analysis[i, aidx] = a[0]
        # a[0]是个什么东西

# Prepare the grid as rectangular list
grid = [
    [analysis[i, j] for j in range(analysis.shape[1])] for i in range(analysis.shape[0])
]
# # Prepare the labels
# label, presm, prob, pred = zip(*text)
# row_labels_left = [
#     ("label: {}".format(label[i]), "pred: {}".format(pred[i]))
#     for i in range(len(label))
# ]
# row_labels_right = [
#     ("logit: {}".format(presm[i]), "prob: {}".format(prob[i]))
#     for i in range(len(label))
# ]
# col_labels = ["".join(method[3]) for method in methods]
#
# # Plot the analysis.
# eutils.plot_image_grid(
#     grid,
#     # row_labels_left,
#     # row_labels_right,
#     col_labels,
#     file_name=os.environ.get("plot_file_name", None),
# )

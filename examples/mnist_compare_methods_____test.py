import warnings
warnings.simplefilter("ignore")
import imp
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as kbackend
import tensorflow.keras.models
tf.compat.v1.disable_eager_execution()
import src.innvestigate as innvestigate
import src.innvestigate.utils as iutils
# Use utility libraries to focus on relevant iNNvestigate routines.
import utils as eutils
import utils.mnist as mnistutils
import matplotlib.pyplot as plot

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

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

scores = mnistutils.train_model(model, data, batch_size=128, epochs=3)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))

# Scale to [0, 1] range for plotting.
def input_postprocessing(X):
    return revert_preprocessing(X) / 255


noise_scale = (input_range[1] - input_range[0]) * 0.1
ri = input_range[0]  # reference input

# Configure analysis methods and properties
# fmt: off
methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN            TITLE
    # Show input
    ("input",               {},                         input_postprocessing,   "Input"),
    # Function
    ("gradient",            {"postprocess": "abs"},     mnistutils.graymap,     "Gradient"),
    ("smoothgrad",
                            {"noise_scale": noise_scale,
                             "postprocess": "square"},  mnistutils.graymap,     "SmoothGrad"),
    # Signal
    ("deconvnet",           {},                         mnistutils.bk_proj,     "Deconvnet"),
    ("guided_backprop",     {},                         mnistutils.bk_proj,     "Guided Backprop"),
    # Interaction
    ("deep_taylor.bounded", {"low": input_range[0],
                             "high": input_range[1]},   mnistutils.heatmap,     "DeepTaylor"),
    ("input_t_gradient",    {},                         mnistutils.heatmap,     "Input * Gradient"),
    ("integrated_gradients",{"reference_inputs": ri},   mnistutils.heatmap,     "Integrated Gradients"),
    ("lrp.z",               {},                         mnistutils.heatmap,     "LRP-Z"),
    ("lrp.epsilon",         {"epsilon": 1},             mnistutils.heatmap,     "LRP-Epsilon"),
]
# fmt: on

# Create model without trailing softmax
model_wo_softmax = innvestigate.model_wo_softmax(model)
# Create analyzers.
analyzers = [
    innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])
    for method in methods
]

analyzer = innvestigate.create_analyzer(
    "lrp.z", model_wo_softmax
)

n = 10
test_images = list(zip(data[2][:n], data[3][:n]))

analysis = np.zeros([len(test_images), len(analyzers), 28, 28, 3])
text = []


for i, (x, y) in enumerate(test_images):  # i是数据序号，x是照片，y是label
    # Add batch axis.
    x = x[None, :, :, :]

    # Predict final activations, probabilites, and label.
    presm = model_wo_softmax.predict_on_batch(x)[0]
    prob = model.predict_on_batch(x)[0]
    y_hat = prob.argmax()

    # Save prediction info:
    text.append(
        (
            "%s" % label_to_class_name[y],  # ground truth label
            "%.2f" % presm.max(),  # pre-softmax logits
            "%.2f" % prob.max(),  # probabilistic softmax output
            "%s" % label_to_class_name[y_hat],  # predicted label
        )
    )

    for aidx, analyzer in enumerate(analyzers):
        # Analyze.
        a = analyzer.analyze(x)
        # Apply common postprocessing, e.g., re-ordering the channels for plotting. 调整channels的顺序
        a = mnistutils.postprocess(a)
        # Apply analysis postprocessing, e.g., creating a heatmap.
        a = methods[aidx][2](a)
        # Store the analysis. a是（2，38，38，1） a[0]是照片（28，28，1）a[1]是len 1
        analysis[i, aidx] = a[0]
        plot.imshow(a[0], cmap="seismic", interpolation="nearest")
        plot.show()
    # a = analyzer.analyze(x)
    # # Apply common postprocessing, e.g., re-ordering the channels for plotting. 调整channels的顺序
    # a = mnistutils.postprocess(a)
    # # Apply analysis postprocessing, e.g., creating a heatmap.
    # a = mnistutils.heatmap(a)
    # plot.imshow(a[0], cmap="seismic", interpolation="nearest")
    # plot.show()

# # Prepare the grid as rectengular list
# grid = [
#     [analysis[i, j] for j in range(analysis.shape[1])] for i in range(analysis.shape[0])
# ]
# b = grid[9][9]
# plot.imshow(grid[9][9], cmap="seismic", interpolation="nearest")
# plot.show()
# # Prepare the labels
# label, presm, prob, pred = zip(*text)
# row_labels_left = [
#     (f"label: {label[i]}", f"pred: {pred[i]}") for i in range(len(label))
# ]
# row_labels_right = [
#     (f"logit: {presm[i]}", f"prob: {prob[i]}") for i in range(len(label))
# ]
# col_labels = ["".join(method[3]) for method in methods]
#
# # Plot the analysis.
# eutils.plot_image_grid(
#     grid,
#     row_labels_left,
#     row_labels_right,
#     col_labels,
#     file_name=os.environ.get("PLOTFILENAME", None),
# )



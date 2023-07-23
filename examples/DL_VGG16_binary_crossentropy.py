import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tempfile
import tensorflow_hub as hub
import matplotlib.pyplot as plot

from keras.applications import VGG16
from datetime import datetime
from keras.preprocessing import image
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from tensorflow.keras import layers
import src.innvestigate as innvestigate
from PIL import Image

from tool import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

tf.compat.v1.disable_eager_execution()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
print("TF version:", tf.__version__)

df = pd.read_csv("file/label_special_form.csv")
# Get label frequencies in descending order
label_freq = df['all_nouns'].apply(lambda s: str(s).split(',')).explode().value_counts().sort_values(ascending=False)

# # Bar plot
# style.use("fivethirtyeight")
# plt.figure(figsize=(12, 10))
# sns.barplot(y=label_freq.index.values, x=label_freq, order=label_freq.index)
# plt.title("Label frequency", fontsize=14)
# plt.xlabel("")
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# Create a list of rare labels 只是要过一遍这个流程，不然shape会不对
rare = list(label_freq[label_freq < 2].index)
print("We will be ignoring these rare labels:", rare)

# Transform all_nouns into a list of labels and remove the rare ones
df['all_nouns'] = df['all_nouns'].apply(lambda s: [l for l in str(s).split(',') if l not in rare])
print(df.head())
print("Number of sample:", len(df))

# 分成训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(df['stop_frame'], df['all_nouns'], test_size=0.2, random_state=44)
print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_val))

# 处理图片数据，把每个图片的路径前面都加上data/使路径变得完整
X_train = [os.path.join('data', str(f)) for f in X_train]
X_val = [os.path.join('data', str(f)) for f in X_val]
print("X_train[:8]:", X_train[:8])

# 处理标签数据
y_train = list(y_train)
y_val = list(y_val)
print("y_train[:8]:", y_train[:8])


# nobs = 8  # Maximum number of images to display
# ncols = 4  # Number of columns in display
# nrows = nobs//ncols  # Number of rows in display

# style.use("default")
# plt.figure(figsize=(12, 4*nrows))
# for i in range(nrows*ncols):
#     ax = plt.subplot(nrows, ncols, i+1)
#     plt.imshow(Image.open(X_train[i]))
#     plt.title(y_train[i], size=10)
#     plt.axis('off')
# plt.show()

# Fit the multi-label binarizer on the training set 在训练集上拟合多标签二值化器
print("Labels:")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))

# 用mlb处理train和val的标签数据，将其转换成二进制的向量,格式为一维数组
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)
print("y_train_bin.shape:", y_train_bin.shape)
print("y_val_bin.shape:", y_val_bin.shape)

# Print example of movie posters and their binary targets
for i in range(3):
    print(X_train[i], y_train_bin[i])

IMG_SIZE = 100  # 100
CHANNELS = 3  # Keep RGB color channels to match the input format of the model


def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector 解码后的图像数据是一个张量
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


BATCH_SIZE = 8  # 8
AUTOTUNE = tf.data.experimental.AUTOTUNE  # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 1024  # Shuffle the training data by a chunck of 1024 observations


def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel 同时解析和预处理观察结果
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle 洗牌 the data each buffer size 缓冲区大小
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training. 在模型训练时，同时在后台预取批次数据，以加快训练过程中的数据加载速度
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)

# for f, l in train_ds.take(1):
#     print("Shape of features array:", f.numpy().shape)
#     print("Shape of labels array:", l.numpy().shape)

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
conv_base.summary()
conv_base.trainable = False

# feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
# feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
#                                          input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
# feature_extractor_layer.trainable = False

# @tf.function
# def macro_soft_f1(y, y_hat):
#     """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
#     Use probability values instead of binary predictions.
#
#     Args:
#         y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
#         y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
#
#     Returns:
#         cost (scalar Tensor): value of the cost function for the batch
#     """
#     y = tf.cast(y, tf.float32)
#     y_hat = tf.cast(y_hat, tf.float32)
#     tp = tf.reduce_sum(y_hat * y, axis=0)
#     fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
#     fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
#     soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
#     cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
#     macro_cost = tf.reduce_mean(cost)  # average on all labels
#     return macro_cost


@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


LR = 1e-5  # Keep it small when transfer learning
EPOCHS = 10
model_bce = tf.keras.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(1024, activation='relu', name='hidden_layer'),
    layers.Dense(N_LABELS, activation='sigmoid', name='output')
])

model_bce.summary()

model_bce.compile(
    optimizer=tf.keras.optimizers.Adam(lr=5e-4),
    loss=tf.keras.metrics.binary_crossentropy,
    metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="DL_VGG16_binary_crossentropy.keras",
        save_best_only=True,
        monitor="val_loss")
]
history_bce = model_bce.fit(train_ds,
                            epochs=EPOCHS,
                            validation_data=create_dataset(X_val, y_val_bin),
                            callbacks=callbacks)
# model_bce_losses, model_bce_val_losses, model_bce_macro_f1s, model_bce_val_macro_f1s = learning_curves(history_bce)
# print("Macro soft-F1 loss: %.2f" %model_bce_val_losses[-1])
# print("Macro F1-score: %.2f" %model_bce_val_macro_f1s[-1])
# y_hat_val = model_bce.predict(train_ds)
# print(y_hat_val)




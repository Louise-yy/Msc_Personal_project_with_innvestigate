# 生成threshold.csv  把test data set的数据个数改为了2，用的是1862个train dataset
import os
import warnings
import logging

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

# 前期设定
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

tf.compat.v1.disable_eager_execution()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
print("TF version:", tf.__version__)

# 开始处理所有的数据
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
X_train, X_val, y_train, y_val = train_test_split(df['stop_frame'], df['all_nouns'], test_size=0.001, random_state=44)
print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_val))
# 把每个图片的路径前面都加上data/使路径变得完整
X_train = [os.path.join('data', str(f)) for f in X_train]
X_val = [os.path.join('data', str(f)) for f in X_val]
print("X_train[:8]:", X_train[:8])
# 把标签数据变成list的格式
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
# 构建一个mlb实例
mlb = MultiLabelBinarizer()
mlb.fit(y_train)
# 将label从string映射成数字

print("Labels:")
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

IMG_SIZE = 100  # Specify height and width of image to match the input format of the model
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


BATCH_SIZE = 8  # Big enough to measure an F1-score
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

# 导入训练好的模型
model_bce = tf.keras.models.load_model("DL_no_macrof1.keras")

# Get all label names
label_names = mlb.classes_
# Performance table with the second model (binary cross-entropy loss)
grid_bce = perf_grid(train_ds, y_train_bin, label_names, model_bce)
print(grid_bce.head(20))
# grid_bce.to_csv('file/grid_bce.csv', index=False)

# Get the maximum F1-score for each label when using the second model and varying the threshold
# 按照'id', 'label', 'freq'进行分组，计算'f1'列的最大值
max_perf = grid_bce.groupby(['id', 'label', 'freq'])[['f1']].max()
# 按照'f1'列进行降序排序
max_perf = max_perf.sort_values('f1', ascending=False)
# 重新设置索引
max_perf = max_perf.reset_index()
# 将'f1'列重命名为'f1max_bce'
max_perf.rename(columns={'f1': 'f1max_bce'}, inplace=True)
# 应用颜色渐变样式，使用sns.light_palette("lightgreen", as_cmap=True)设置颜色映射
max_perf.style.background_gradient(subset=['freq', 'f1max_bce'], cmap=sns.light_palette("lightgreen", as_cmap=True))
print(max_perf)

# Get the maximum F1-score for each label when using the second model and varying the threshold
# 根据'id'、'label'和'freq'列进行分组，并找到每个组内'f1'值最大的那一行
max_f1_rows = grid_bce.groupby(['id', 'label', 'freq'])['f1'].idxmax()
# 通过索引获取具有最大'f1'值的行
result = grid_bce.loc[max_f1_rows]
# 按照'f1'列进行降序排序
max_perf = result.sort_values('f1', ascending=False)
print(max_perf)
# max_perf.to_csv('file/threshold_new.csv', index=False)


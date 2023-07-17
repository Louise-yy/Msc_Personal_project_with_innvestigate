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
df = pd.read_csv("file/label_special_after_change.csv")
# Get label frequencies in descending order
label_freq = df['all_nouns'].apply(lambda s: str(s).split(',')).explode().value_counts().sort_values(ascending=False)
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
# 构建一个mlb实例
mlb = MultiLabelBinarizer()
# 将label从string映射成数字
mlb.fit(y_train)

print("Labels:")
# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))

# 用mlb处理标签数据，将其转换成二进制的向量,格式为一维数组
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)

IMG_SIZE = 100  # Specify height and width of image to match the input format of the model
CHANNELS = 3  # Keep RGB color channels to match the input format of the model


# 从测试集中取第一个图片出来，对它进行处理
filename = X_val[0]
# Read an image from a file
image_string = tf.io.read_file(filename)
# Decode it into a dense vector 解码后的图像数据是一个张量
image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
# Resize it to fixed shape
image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
# Normalize it from [0, 255] to [0.0, 1.0]
image_normalized = image_resized / 255.0
# 把图片的shape从(100,100,3)转换为(1,100,100,3)是为了符合模型的输入形状
image_normalized_expanded = tf.expand_dims(image_normalized, axis=0)

# 展示测试集的第一张图片
print("filename: ", filename)
image = Image.open(filename)
image_resized = image.resize((100, 100))
# image_resized.show()
plt.imshow(image_resized, cmap="gray", interpolation="nearest")
plt.show()


# 导入训练好的模型
model_bce = tf.keras.models.load_model("DL_no_macrof1.keras")


# Stripping the softmax activation from the model
model_wo_sm = innvestigate.model_wo_softmax(model_bce)

# Creating analyzer
gradient_analyzer = innvestigate.create_analyzer("gradient", model_wo_sm)
# Applying the analyzer
analysis = gradient_analyzer.analyze(image_normalized_expanded)
# Displaying the gradient
analysis_add = analysis + 1
# # 找到图像数据的最小值和最大值
# min_value = np.min(analysis_add)
# max_value = np.max(analysis_add)
# # 将图像数据映射到0-1的区间内
# analysis_normalized = ((analysis_add - min_value) / (max_value - min_value))
# analysis_255 = analysis_normalized * 255
plot.imshow(analysis_add.squeeze(), cmap="seismic", interpolation="nearest")
plot.show()

# # Creating a parameterized analyzer
# abs_gradient_analyzer = innvestigate.create_analyzer(
#     "gradient", model_wo_sm, postprocess="abs"
# )
# square_gradient_analyzer = innvestigate.create_analyzer(
#     "gradient", model_wo_sm, postprocess="square"
# )
# # Applying the analyzers
# abs_analysis = abs_gradient_analyzer.analyze(image_normalized_expanded)
# square_analysis = square_gradient_analyzer.analyze(image_normalized_expanded)
# # Displaying the analyses, use gray map as there no negative values anymore
# # abs_analysis_add = abs_analysis + 1
# # square_analysis_add = square_analysis + 1
# plot.imshow(abs_analysis.squeeze(), cmap="gray", interpolation="nearest")
# plot.show()
# plot.imshow(square_analysis.squeeze(), cmap="gray", interpolation="nearest")
# plot.show()


# Creating an analyzer and set neuron_selection_mode to "index"
inputXgradient_analyzer = innvestigate.create_analyzer(
    "input_t_gradient", model_wo_sm, neuron_selection_mode="index"
)
for neuron_index in range(20):
    print("Analysis w.r.t. to neuron", neuron_index)
    # Applying the analyzer and pass that we want
    analysis = inputXgradient_analyzer.analyze(image_normalized_expanded, neuron_index)

    # Displaying the gradient
    analysis_add = analysis + 1
    plot.imshow(analysis_add.squeeze(), cmap="seismic", interpolation="nearest")
    plot.show()

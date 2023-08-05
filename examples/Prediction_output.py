# 得到prediction.csv
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

# Create a list of rare labels 只是要过一遍这个流程，不然shape会不对
rare = list(label_freq[label_freq < 2].index)
print("We will be ignoring these rare labels:", rare)
# Transform all_nouns into a list of labels and remove the rare ones
df['all_nouns'] = df['all_nouns'].apply(lambda s: [l for l in str(s).split(',') if l not in rare])
print(df.head())
print("Number of sample:", len(df))
# 分成训练集和测试集
# X_train, X_val, y_train, y_val = train_test_split(df['stop_frame'], df['all_nouns'], test_size=0.2, random_state=44)
X_val = df['stop_frame']
y_val = df['all_nouns']
# print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_val))
# 把每个图片的路径前面都加上data/使路径变得完整
# X_train = [os.path.join('data', str(f)) for f in X_train]
X_val = [os.path.join('data', str(f)) for f in X_val]
print("X_train[:8]:", X_val[:8])
# 把标签数据变成list的格式
# y_train = list(y_train)
y_val = list(y_val)
print("y_train[:8]:", y_val[:8])

# Fit the multi-label binarizer on the training set 在训练集上拟合多标签二值化器
# 构建一个mlb实例
mlb = MultiLabelBinarizer()
mlb.fit(y_val)
# 将label从string映射成数字

print("Labels:")
# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))

# 用mlb处理train和val的标签数据，将其转换成二进制的向量,格式为一维数组
# y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)
# print("y_train_bin.shape:", y_train_bin.shape)
print("y_val_bin.shape:", y_val_bin.shape)

# Print example of movie posters and their binary targets
for i in range(3):
    print(X_val[i], y_val_bin[i])

IMG_SIZE = 200  # Specify height and width of image to match the input format of the model
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


BATCH_SIZE = 2  # Big enough to measure an F1-score
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


# train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)

# 导入训练好的模型
model_bce = tf.keras.models.load_model("DL_VGG16_binary_crossentropy_200.keras")

# # 选择图片
# img_path = X_val[0]
# labels = y_val[0]
# # Read and prepare image
# img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE, CHANNELS))
# plt.imshow(img)
# plt.show()
# img = image.img_to_array(img)
# img = img / 255
# img = np.expand_dims(img, axis=0)
# predict = model_bce.predict(img)

# im = Image.open(img_path)


# Generate prediction
# prediction = (predict > 0.5).astype('int')
d = pd.read_csv("file/threshold_10000_200.csv")
threshold = d['threshold']
output_folder = "output"


def get_per_prediction(img_path):
    print("img_path: ", img_path)
    # i = image.load_img(img_path, target_size=(200, 200, CHANNELS))
    # i.save(os.path.join(output_folder, img_path))
    # print(f"imgae saved to：{os.path.join(output_folder, img_path)}")
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE, CHANNELS))
    img = image.img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    predict = model_bce.predict(img)
    # print("predict number:", predict)
    per_img_prediction = []
    for i in range(20):
        v = predict[0, i]
        t = d.loc[d['id'] == i, 'threshold'].values[0]
        if v >= t:
            per_img_prediction.append(int(1))
        else:
            per_img_prediction.append(int(0))
    prediction_boolean_list.append(per_img_prediction)
    # print("predict boolean:", per_img_prediction)

    per_img_prediction = pd.Series(per_img_prediction)
    per_img_prediction.index = mlb.classes_
    per_img_prediction_noun = per_img_prediction[per_img_prediction == 1].index.values
    # print("predict noun:", per_img_prediction_noun)
    return predict, per_img_prediction_noun


# # 这只是得到了一个图片的prediction
# predict_number, predict_boolean, predict_noun = get_per_prediction(X_val[1])
# print(test)

# 获取整个test dataset的prediction
prediction_number_list = []
prediction_boolean_list = []
prediction_noun_list = []
for x in X_val:
    predict_number, predict_noun = get_per_prediction(x)
    prediction_number_list.append(predict_number)
    prediction_noun_list.append(predict_noun)

# print(prediction_number_list)
# print(prediction_boolean_list)
# print(prediction_noun_list)

p_number_1 = []
p_number_2 = []
p_number_3 = []
p_number_4 = []
p_number_5 = []
p_number_6 = []
p_number_7 = []
p_number_8 = []
p_number_9 = []
p_number_10 = []
p_number_11 = []
p_number_12 = []
p_number_13 = []
p_number_14 = []
p_number_15 = []
p_number_16 = []
p_number_17 = []
p_number_18 = []
p_number_19 = []
p_number_20 = []
for a in prediction_number_list:
    p_number_1.append(a[0, 0])
    p_number_2.append(a[0, 1])
    p_number_3.append(a[0, 2])
    p_number_4.append(a[0, 3])
    p_number_5.append(a[0, 4])
    p_number_6.append(a[0, 5])
    p_number_7.append(a[0, 6])
    p_number_8.append(a[0, 7])
    p_number_9.append(a[0, 8])
    p_number_10.append(a[0, 9])
    p_number_11.append(a[0, 10])
    p_number_12.append(a[0, 11])
    p_number_13.append(a[0, 12])
    p_number_14.append(a[0, 13])
    p_number_15.append(a[0, 14])
    p_number_16.append(a[0, 15])
    p_number_17.append(a[0, 16])
    p_number_18.append(a[0, 17])
    p_number_19.append(a[0, 18])
    p_number_20.append(a[0, 19])

y_val_str = [','.join(map(str, p_list)) for p_list in y_val]
y_val_bin_str = [' '.join(map(str, p_list)) for p_list in y_val_bin]
# m = [str(threshold[0]) for _ in range(37)]
# 创建 DataFrame
dataf = pd.DataFrame({
    'frame': X_val,
    # 'predict_number': prediction_number_list,
    # 'predictive_boolean': prediction_boolean_list,
    # 'label_boolean': y_val_bin_str,
    'predictive_noun': prediction_noun_list,
    'label_noun': y_val_str,
    'bowl': p_number_1,
    'bowl_threshold': [str(d.loc[d['id'] == 0, 'threshold'].values[0]) for _ in range(len(X_val))],
    'cloth': p_number_2,
    'cloth_threshold': [str(d.loc[d['id'] == 1, 'threshold'].values[0]) for _ in range(len(X_val))],
    'container': p_number_3,
    'container_threshold': [str(d.loc[d['id'] == 2, 'threshold'].values[0]) for _ in range(len(X_val))],
    'cupboard': p_number_4,
    'cupboard_threshold': [str(d.loc[d['id'] == 3, 'threshold'].values[0]) for _ in range(len(X_val))],
    'dough': p_number_5,
    'dough_threshold': [str(d.loc[d['id'] == 4, 'threshold'].values[0]) for _ in range(len(X_val))],
    'drawer': p_number_6,
    'drawer_threshold': [str(d.loc[d['id'] == 5, 'threshold'].values[0]) for _ in range(len(X_val))],
    'fork': p_number_7,
    'fork_threshold': [str(d.loc[d['id'] == 6, 'threshold'].values[0]) for _ in range(len(X_val))],
    'fridge': p_number_8,
    'fridge_threshold': [str(d.loc[d['id'] == 7, 'threshold'].values[0]) for _ in range(len(X_val))],
    'glass': p_number_9,
    'glass_threshold': [str(d.loc[d['id'] == 8, 'threshold'].values[0]) for _ in range(len(X_val))],
    'hand': p_number_10,
    'hand_threshold': [str(d.loc[d['id'] == 9, 'threshold'].values[0]) for _ in range(len(X_val))],
    'knife': p_number_11,
    'knife_threshold': [str(d.loc[d['id'] == 10, 'threshold'].values[0]) for _ in range(len(X_val))],
    'lid': p_number_12,
    'lid_threshold': [str(d.loc[d['id'] == 11, 'threshold'].values[0]) for _ in range(len(X_val))],
    'meat': p_number_13,
    'meat_threshold': [str(d.loc[d['id'] == 12, 'threshold'].values[0]) for _ in range(len(X_val))],
    'onion': p_number_14,
    'onion_threshold': [str(d.loc[d['id'] == 13, 'threshold'].values[0]) for _ in range(len(X_val))],
    'pan': p_number_15,
    'pan_threshold': [str(d.loc[d['id'] == 14, 'threshold'].values[0]) for _ in range(len(X_val))],
    'plate': p_number_16,
    'plate_threshold': [str(d.loc[d['id'] == 15, 'threshold'].values[0]) for _ in range(len(X_val))],
    'spatula': p_number_17,
    'spatula_threshold': [str(d.loc[d['id'] == 16, 'threshold'].values[0]) for _ in range(len(X_val))],
    'sponge': p_number_18,
    'sponge_threshold': [str(d.loc[d['id'] == 17, 'threshold'].values[0]) for _ in range(len(X_val))],
    'spoon': p_number_19,
    'spoon_threshold': [str(d.loc[d['id'] == 18, 'threshold'].values[0]) for _ in range(len(X_val))],
    'tap': p_number_20,
    'tap_threshold': [str(d.loc[d['id'] == 19, 'threshold'].values[0]) for _ in range(len(X_val))]
})

print(dataf)

# 保存 DataFrame 到 'result.csv'
dataf.to_csv('output/prediction.csv', index=False)

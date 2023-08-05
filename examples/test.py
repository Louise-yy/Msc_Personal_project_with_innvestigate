# threshold main!!!! 用的全部数据，且threshold精确到了小数点后三位
import os
import numpy as np
import pandas as pd
import warnings
import logging
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

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
# X_train, X_val, y_train, y_val = train_test_split(df['stop_frame'], df['all_nouns'], test_size=0.001, random_state=44)
X_train = df['stop_frame']
y_train = df['all_nouns']
print("Number of posters for training: ", len(X_train))
# print("Number of posters for validation: ", len(X_val))
# 把每个图片的路径前面都加上data/使路径变得完整
X_train = [os.path.join('data', str(f)) for f in X_train]
# X_val = [os.path.join('data', str(f)) for f in X_val]
print("X_train[:8]:", X_train[:8])
# 把标签数据变成list的格式
y_train = list(y_train)
# y_val = list(y_val)
print("y_train[:8]:", y_train[:8])

# 构建一个mlb实例
mlb = MultiLabelBinarizer()
# 将label从string映射成数字
mlb.fit(y_train)

print("Labels:")
# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))

# 用mlb处理train和val的标签数据，将其转换成二进制的向量,格式为一维数组
y_train_bin = mlb.transform(y_train)
# y_val_bin = mlb.transform(y_val)
print("y_train_bin.shape:", y_train_bin.shape)
# print("y_val_bin.shape:", y_val_bin.shape)

# Print example of movie posters and their binary targets
for i in range(3):
    print(X_train[i], y_train_bin[i])

IMG_SIZE = 200  # ############1
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


BATCH_SIZE = 2  # #################2
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
# val_ds = create_dataset(X_val, y_val_bin)


@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost

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


# ###############3
# 定义KerasLayer的包装函数
def KerasLayerWrapper(*args, **kwargs):
    return hub.KerasLayer(*args, **kwargs)

# 加载模型时使用custom_objects参数来传递自定义层的定义
# model_bce = tf.keras.models.load_model("DL_mobilenetV2_macro_soft_f1.keras", custom_objects={"KerasLayer": KerasLayerWrapper, "macro_soft_f1": macro_soft_f1, "macro_f1": macro_f1})

# # 导入训练好的模型
model_bce = tf.keras.models.load_model("DL_VGG16_binary_crossentropy_200.keras")

def perf_grid(ds, target, label_names, model, n_thresh=10000):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn,
    precision, recall and f-score metrics for each label.

    Args:
        ds (tf.data.Datatset): contains the features array
        target (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        label_names (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int) : number of thresholds to try

    Returns:
        grid (Pandas dataframe): performance table
    """

    # Get predictions
    y_hat_val = model.predict(ds)
    # Define target matrix
    y_val = target
    # Find label frequencies in the validation set
    label_freq = target.sum(axis=0)
    # Get label indexes
    label_index = [i for i in range(len(label_names))]
    # Define thresholds
    thresholds = np.linspace(0, 1, n_thresh + 1).astype(np.float32)

    # Compute all metrics for all labels
    ids, labels, freqs, tps, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], [], []
    for l in label_index:
        for thresh in thresholds:
            ids.append(l)
            labels.append(label_names[l])
            f = round(label_freq[l] / len(y_val), 2)
            freqs.append(f)
            y_hat = y_hat_val[:, l]  # prediction数值
            y = y_val[:, l]  # label
            y_pred = y_hat > thresh  # 是一个boolean值
            tp = np.count_nonzero(y_pred * y)
            fp = np.count_nonzero(y_pred * (1 - y))
            fn = np.count_nonzero((1 - y_pred) * y)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Create the performance dataframe
    grid = pd.DataFrame({
        'id': ids,
        'label': labels,
        'freq': freqs,
        'threshold': list(thresholds) * len(label_index),
        'tp': tps,
        'fp': fps,
        'fn': fns,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s})

    grid = grid[['id', 'label', 'freq', 'threshold',
                 'tp', 'fn', 'fp', 'precision', 'recall', 'f1']]

    return grid

# Get all label names
label_names = mlb.classes_
# Performance table with the second model (binary cross-entropy loss)
grid_bce = perf_grid(train_ds, y_train_bin, label_names, model_bce)
print(grid_bce.head(20))
grid_bce.to_csv('file/grid_bce.csv', index=False)


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
max_perf.to_csv('file/threshold_10000_200.csv', index=False)
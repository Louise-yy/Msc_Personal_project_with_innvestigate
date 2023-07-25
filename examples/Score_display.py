import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

# 读取
file_path = 'output/prediction.csv'
df = pd.read_csv(file_path)

# 提取指定行
target_row = df[df['frame'] == 'data/P01_102_10053.jpg']

# 取出列的数值并进行排序
data_values = target_row[['bowl', 'cloth', 'container', 'cupboard', 'dough',
                          'drawer', 'fork', 'fridge', 'glass', 'hand',
                          'knife', 'lid', 'meat', 'onion', 'pan',
                          'plate', 'spatula', 'sponge', 'spoon', 'tap']].values.flatten()
sorted_values = sorted(data_values)

# 取排序后的前十个值
top_10_values = sorted_values[10:]

# 对应的列名
categorys = []
for number in top_10_values:
    # 从目标行中查找包含该数值的列名
    # column = target_row.columns
    for col in target_row.columns:
        # value = target_row[col].values
        if target_row[col].values == number:
            categorys.append(col)
            break

# 创建新的dataframe
data = pd.DataFrame({'Category': categorys,
                     'Value': top_10_values})
print(data)
# 读取file2.csv
file2_path = 'file/threshold_10000.csv'
df2 = pd.read_csv(file2_path)

# # 合并两个dataframe并提取threshold列
# merged_df = pd.merge(new_df, df2[['label', 'threshold']], left_on='Category', right_on='label')
# threshold_values = merged_df['threshold'].values

thresholds = []
for category in categorys:
    target_row = df2[df2['label'] == category]
    target_data = target_row['threshold'].values[0]
    thresholds.append(target_data)
print(thresholds)

# 假设你有以下数据
# data = {'Category': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
#         'Value': [10, 15, 7, 20, 12, 11, 10, 15, 2, 7]}
# df = pd.DataFrame(data)

# 添加红线对应的数值
# red_lines = [2.5, 5, 7.5, 15, 10, 11, 7, 8, 9, 10]

plt.figure(figsize=(16, 10))

# 绘制横向条形图
plt.barh(data['Category'], data['Value'])
# plt.xlim(0.0, 10.0)
plt.ylim(-0.5, len(data) - 0.5)

# 在每个条形上添加垂直的红线
for i, red_line in enumerate(thresholds):
    ymin = 0.05 + i * 0.1 - 0.05
    ymax = 0.05 + i * 0.1 + 0.05
    plt.axvline(x=red_line, ymin=ymin, ymax=ymax, color='red', linewidth=2)

# 设置x轴标签、y轴标签和图表标题
# plt.xlabel('Value')
# plt.ylabel('Category')
# plt.title('Horizontal Bar Chart')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.show()

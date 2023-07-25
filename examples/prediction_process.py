# 取prediction.csv中的跟cupboard和pan有关的数据
import pandas as pd

# 读取CSV文件
data = pd.read_csv("output/prediction.csv")

# 处理label_noun列的数据
data["label_noun"] = data["label_noun"].apply(lambda x: x.split(","))

# 将predictive_noun和label_noun列中含有'cupboard'和'pan'的行提取出来
filtered_data = data[(data['predictive_noun'].str.contains('cupboard') | data['predictive_noun'].str.contains('pan')) |
                 (data['label_noun'].str.contains('cupboard') | data['label_noun'].str.contains('pan'))]

# 显示处理后的数据
print(filtered_data)
filtered_data.to_csv('output/prediction_cupboard_pan.csv', index=False)
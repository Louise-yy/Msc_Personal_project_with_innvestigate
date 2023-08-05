import pandas as pd
import re

# # 取prediction.csv中的跟cupboard和pan有关的数据
# # 读取CSV文件
# data = pd.read_csv("output/prediction.csv")
#
# # 处理label_noun列的数据
# data["label_noun"] = data["label_noun"].apply(lambda x: x.split(","))
#
# # 将predictive_noun和label_noun列中含有'cupboard'和'pan'的行提取出来
# filtered_data = data[(data['predictive_noun'].str.contains('cupboard') | data['predictive_noun'].str.contains('pan')) |
#                  (data['label_noun'].str.contains('cupboard') | data['label_noun'].str.contains('pan'))]
#
# # 显示处理后的数据
# print(filtered_data)
# filtered_data.to_csv('output/prediction_cupboard_pan.csv', index=False)



# 取prediction_cupboard_pan.csv中的FN, 有但没识别出来
data = pd.read_csv('output/prediction.csv')

# 将CSV数据转换为集合形式，以便进行高效的成员检查
predictive_noun_set = data['predictive_noun']
label_noun_set = data['label_noun']

# 筛选数据，保留'label_noun'列中至少有一个单词不在'predictive_noun_set'集合中的行
filtered_data = []
for i in range(len(predictive_noun_set)):
    prediction = predictive_noun_set[i]
    # 去除首尾的方括号，并将所有单词之间的空格替换为逗号
    # prediction_str = prediction.strip("[]\n").replace(" ", ",")
    prediction_str = re.sub(r'\'|\[|\]|\n', '', prediction).replace(" ", ",")
    prediction_array = prediction_str.split(',')

    label = label_noun_set[i]
    label_str = re.sub(r'\'|\[|\]', '', label).replace(" ", "")
    label_array = label_str.split(',')

    if 'drawer' in label_array:
        # 判断label中的单词是否有一个不存在于prediction中
        for word in label_array:
            if word not in prediction_array:
                row = data.iloc[i]
                filtered_data.append(data.iloc[i])
                break
# 打印筛选后的结果
print(filtered_data)

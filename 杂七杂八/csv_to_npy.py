import pandas as pd
import numpy as np

# 读入csv文件
df = pd.read_csv(r'D:\任务归档\电池\研究\data\selected_data\battery1\battery1_cycle.csv', encoding='gbk')        # 如果有表头，默认会读表头；如果没有，可以加header=None

# 转成numpy数组
arr = df.to_numpy()                 # 或 df.values

# 保存为npy文件
np.save(r'D:\任务归档\电池\研究\data\selected_data\battery1\battery1_cycle.npy', arr)

# 读取时：arr2 = np.load('data.npy')

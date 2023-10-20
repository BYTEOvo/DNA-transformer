import numpy as np

# 读取.npy文件
data = np.load('data/eco_TIS/eco_TIS.npy')

# 计算要保留的数据数量
n = int(len(data) * 0.01)

# 仅保留前百分之一的数据
selected_data = data[:n]

# 写入数据到新的.npy文件
output_path = 'newData/new_data.npy'
np.save(output_path, selected_data)

print(f"数据已成功写入新文件：{output_path}")
import numpy as np

# 读取.npy文件
data = np.load('data/eco_regulon/eco_regulon_70.npy')
# a=np.sum(data,axis=0)
# 将数据转换为整数类型
# data = data.astype(int)
print(data)

# 写入数据到新文本文件
output_path = 'newData/new_data_regulon_70.txt'
np.savetxt(output_path, data, fmt='%d')

print(f"数据已成功写入新文件：{output_path}")
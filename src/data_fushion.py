import numpy as np

#
# # 从txt文件中加载相似性矩阵
KM1 = np.loadtxt('D:\yjs\DAF-LA\data\mg.txt')
KM2 = np.loadtxt('D:\yjs\DAF-LA\data\mja.txt')
KM3 = np.loadtxt('D:\yjs\DAF-LA\data\mfs.txt')

# 逐元素取最大值
result_matrix = np.maximum(np.maximum(KM1, KM2), KM3)

# 将结果保存到 txt 文件
np.savetxt('D:\yjs\DAF-LA\data/mn.txt', result_matrix,fmt='%.6f', delimiter='\t')

print("结果已保存到 result_matrix.txt 文件中")
KM1 = np.loadtxt('D:\yjs\DAF-LA\data\dg.txt')
KM2 = np.loadtxt('D:\yjs\DAF-LA\data\dja.txt')
KM3 = np.loadtxt('D:\yjs\DAF-LA\data\DisSim.txt')
result_matrix = np.maximum(np.maximum(KM1, KM2), KM3)
np.savetxt('D:\yjs\DAF-LA\data/dn.txt', result_matrix,fmt='%.6f', delimiter='\t')
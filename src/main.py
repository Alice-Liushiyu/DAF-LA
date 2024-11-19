
import numpy as np

# 从txt文件中读取代谢物特征矩阵
import pandas as pd
from keras.layers import Dropout
from keras.optimizer_v2.adam import Adam
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import tensorflow as tf

from sklearn.model_selection import cross_val_score, cross_validate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_curve, auc, precision_recall_curve, matthews_corrcoef, accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score

k = 10
np.random.seed(40)
tf.random.set_seed(40)

metabolite_matrix = np.loadtxt(f'D:\yjs\DAF-LA\data\meta_features_{k}.txt')
# 从txt文件中读取疾病特征矩阵

disease_matrix = np.loadtxt(f'D:\yjs\DAF-LA\data\dis_features_{k}.txt')

label = np.loadtxt('D:\yjs\DAF-LA\data\dm.txt')

num_metabolites = metabolite_matrix.shape[0]
num_disease = disease_matrix.shape[0]
F_matrix = np.zeros((num_metabolites, num_disease,256))




m = np.hstack((metabolite_matrix[0], disease_matrix[0]))

F_matrix[0][0]=m

for i in range(num_metabolites):
    for j in range(num_disease):
        m=np.hstack((metabolite_matrix[i], disease_matrix[j]))
        F_matrix[i][j]=m

reshaped_features = F_matrix.reshape(1435 * 177, -1)



flattened_adjacency_matrix = label.reshape(-1)#
print(flattened_adjacency_matrix)
sum=0
for i in flattened_adjacency_matrix:
    sum=sum+i
print(sum)


X=reshaped_features
print(X.shape)
y=flattened_adjacency_matrix
print(y.shape)

import numpy as np
from collections import Counter


def undersample(X, y):
    # 将特征矩阵和标签合并
    data = np.hstack((X, y.reshape(-1, 1)))

    # 统计每个类别的样本数量
    counter = Counter(y)
    # 找出样本数量最少的类别
    min_class_count = min(counter.values())

    # 对每个类别进行欠采样
    sampled_data = []
    for label in counter:
        # 获取当前类别的样本索引
        label_indices = np.where(data[:, -1] == label)[0]
        # 随机选择与最少类别相同数量的样本
        selected_indices = np.random.choice(label_indices, int(min_class_count/2), replace=True)
        # 将选定的样本添加到采样数据中
        sampled_data.extend(data[selected_indices])

    # 将采样后的数据转换为数组格式
    sampled_data = np.array(sampled_data)

    # 分割特征和标签
    X_sampled = sampled_data[:, :-1]
    y_sampled = sampled_data[:, -1]

    return X_sampled, y_sampled

# 进行欠采样
X_sampled, y_sampled = undersample(X, y)

print("原始标签计数:", Counter(y))
print("欠采样后标签计数:", Counter(y_sampled))
X=X_sampled
y=y_sampled

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import tensorflow as tf
from tensorflow.keras import layers, models


from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # Assuming inputs shape is (batch_size, seq_len, embed_dim)
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention_output, _ = self.attention(query, key, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerAutoencoder(tf.keras.Model):
    def __init__(self, input_size, output_size, embed_dim, num_heads):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            MultiHeadSelfAttention(embed_dim, num_heads),
            # layers.Dense(embed_dim, activation='relu'),
            # layers.Dense(output_size, activation='relu'),
            layers.Dense(output_size, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            MultiHeadSelfAttention(embed_dim, num_heads),
            # layers.Dense(embed_dim, activation='relu'),
            layers.Dense(input_size)
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


sample_matrix = X
label_matrix = y
flod = 5
# 定义五折交叉验证
kfold = KFold(n_splits=flod, shuffle=True,random_state=40)
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tensorflow as tf

auc_scores = []
aupr_scores = []
fprs = []
tprs = []
precisions = []
recalls = []
F1_scores = []
accuracy_scores = []
mcc_scores = []
specificity_scores = []
precisionss = []
recallss = []
threshold = 0.6

for train_index, test_index in kfold.split(sample_matrix):
    # 划分训练集和测试集
    X_train, X_test = sample_matrix[train_index], sample_matrix[test_index]
    y_train, y_test = label_matrix[train_index], label_matrix[test_index]

    # 构建并训练模型
    output_size = 1612
    model = TransformerAutoencoder(input_size=X_train.shape[1], output_size=output_size, embed_dim=128,
                                   num_heads=2)  # 128 2
    # 创建一个优化器，例如Adam优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 设置学习率为0.001

    # 编译模型时传入优化器
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    model.fit(X_train, X_train, epochs=5, batch_size=64, verbose=0)

    X_train_encoded = model.encoder(X_train).numpy()
    X_test_encoded = model.encoder(X_test).numpy()
    # 假设你有一个形状为 (2235, 1, 512) 的三维矩阵

    # 使用 squeeze() 方法移除维度为 1 的维度
    X_train_encoded = np.squeeze(X_train_encoded, axis=1)
    X_test_encoded = np.squeeze(X_test_encoded, axis=1)
 
    X_train_cnn = np.expand_dims(X_train_encoded, axis=-1)
    X_test_cnn = np.expand_dims(X_test_encoded, axis=-1)


    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(1612, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.7),
        Dense(1, activation='sigmoid')
    ])
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)
    # 编译模型
    model.compile(optimizer=optimizer1, loss='binary_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_data=(X_test_cnn, y_test), verbose=0)
    y_pred_proba = model.predict(X_test_cnn)
    # 计算 AUC 值
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # roc_auc = auc(fpr, tpr)
    aucc = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(aucc)
    fprs.append(fpr)
    tprs.append(tpr)
    # 计算 AUPR 值
    # print(auc_scores)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    aupr = average_precision_score(y_test, y_pred_proba)
    # print(aupr)
    aupr_scores.append(aupr)
    precisions.append(precision)
    recalls.append(recall)
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    # 计算acc
    accuracy = accuracy_score(y_test, y_pred_binary)
    accuracy_scores.append(accuracy)
    # 计算 Matthews相关系数
    mcc = matthews_corrcoef(y_test, y_pred_binary)
    mcc_scores.append(mcc)
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    # 计算特异度
    specificity = tn / (tn + fp)
    specificity_scores.append(specificity)
    # 计算recall和pre
    precisionn = precision_score(y_test, y_pred_binary)
    recalll = recall_score(y_test, y_pred_binary)
    precisionss.append(precisionn)
    recallss.append(recalll)
    # 计算f1
    f1 = f1_score(y_test, y_pred_binary)
    F1_scores.append(f1)

from tensorflow.keras.models import load_model

# 假设您的模型已经训练好了，并且已经加载到了变量 model 中

# 保存模型到文件
# model.save("D:\yjs\project\GCNAT-main\Change\model.h5")
# 打印五折交叉验证下的 AUC 值
# print("AUC scores for each fold:", auc_scores)
# print("AUPR scores for each fold:", aupr_scores)
# print("F1 scores for each fold:", F1_scores)
# print("precisions scores for each fold:", precisionss)
# print("recalls scores for each fold:", recallss)
# print("acc scores for each fold:", accuracy_scores)
# print("mcc scores for each fold:", mcc_scores)
# print("specificity scores for each fold:", specificity_scores)
# 打印平均 AUC 值
# print("embed_dim=", ed)
# print("num_head=", j)
print("Mean AUC:", np.mean(auc_scores))
print("Mean AUPR:", np.mean(aupr_scores))
print("Mean F1 scores for each fold:", np.mean(F1_scores))
print("Mean precisions scores for each fold:", np.mean(precisionss))
print("Mean recalls scores for each fold:", np.mean(recallss))
print("Mean acc scores scores for each fold:", np.mean(accuracy_scores))
print("Mean mcc scores scores for each fold:", np.mean(mcc_scores))
print("Mean specificity scores scores for each fold:", np.mean(specificity_scores))
import matplotlib.pyplot as plt
# 计算平均 ROC 曲线
# 计算平均 ROC 曲线
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros_like(mean_fpr)
tprs_interp = []
for fpr, tpr in zip(fprs, tprs):
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
mean_tpr /= flod
mean_tpr[0] = 0.0
# mean_auc = np.mean(auc_scores)
# mean_roc_auc = auc(mean_fpr, mean_tpr)

# 绘制平均 ROC 曲线
# np.savetxt('D:\yjs\project\GCNAT-main\Change2\data/DAF-LA_2kflod_fpr.txt', mean_fpr)
# np.savetxt('D:\yjs\project\GCNAT-main\Change2\data/DAF-LA_2kflod_tpr.txt', mean_tpr)
# plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {np.mean(auc_scores):.4f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Cross Validation ROC Curve')
# plt.legend(loc='lower right')
# plt.show()
#
# # 计算平均 PR 曲线
# mean_precision = np.linspace(0, 1, 100)
# mean_recall = np.mean([np.interp(mean_precision, precision, recall) for precision, recall in zip(precisions, recalls)], axis=0)
# mean_pr_auc = np.mean(auc_scores)
# np.savetxt('D:\yjs\project\GCNAT-main\Change2\data/DAF-LA_2kflod_precision.txt', mean_precision)
# np.savetxt('D:\yjs\project\GCNAT-main\Change2\data/DAF-LA_2kflod_recall.txt', mean_recall)
#
# # 绘制平均 PR 曲线
# plt.plot(mean_recall, mean_precision, color='r', label=f'Mean PR (AUPR = {np.mean(aupr_scores):.4f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Cross Validation PR Curve')
# plt.legend(loc='lower right')
# plt.show()
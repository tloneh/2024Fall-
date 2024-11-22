import pickle
import numpy as np
import os
from sklearn.utils import shuffle
from softmax.softmax import train_softmax
from MLP.MLP import train_MLP
# from MLP.MLP import train_MLP_with_optimizers
from MLP.MLP_opt import train_MLP_with_optimizers
from CNN.CNN import train_CNN

def load_data(dir):
    """
    加载 CIFAR-10 数据集。
    
    参数:
    dir (str): 数据集文件所在的目录路径。
    
    返回:
    tuple: (X_train, Y_train, X_test, Y_test)，分别是训练数据、训练标签、测试数据、测试标签。
    """
    def load_batch(file_path):
        """加载单个批次的 CIFAR-10 数据。"""
        with open(file_path, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            data = data_dict[b'data']
            labels = data_dict[b'labels']
            return data, labels

    # 检查路径是否存在
    if not os.path.exists(dir):
        raise ValueError(f"指定的目录 {dir} 不存在！请检查路径。")
    
    # 加载训练数据
    X_train, Y_train = [], []
    for i in range(1, 6):
        file_path = os.path.join(dir, f'data_batch_{i}')
        data, labels = load_batch(file_path)
        X_train.append(data)
        Y_train += labels
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.array(Y_train)

    # 加载测试数据
    test_file_path = os.path.join(dir, 'test_batch')
    X_test, Y_test = load_batch(test_file_path)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test

# 数据预处理函数
def preprocess_data(X_train, X_test):
    """将图像数据标准化为 [0, 1] 范围，并改为3*32*32。"""
    # CNN使用
    X_train = X_train.reshape(-1, 3, 32, 32) / 255.0
    X_test = X_test.reshape(-1, 3, 32, 32) / 255.0
    # 其他网络使用
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0
    return X_train, X_test



# 主函数
def main():
    # 参数设置
    data_dir = './data'  # CIFAR-10 数据路径
    seed = 42
    # 加载数据
    X_train, Y_train, X_test, Y_test = load_data(data_dir)
    X_train, X_test = preprocess_data(X_train, X_test)

    # 随机打乱训练数据
    X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)# 种子为42
    X_test, Y_test = shuffle(X_test, Y_test, random_state=seed)

    # 选择不同模型进行训练
    # train_softmax(X_train, Y_train, X_test, Y_test)
    # train_MLP(X_train, Y_train, X_test, Y_test)
    # train_MLP_with_optimizers(X_train, Y_train, X_test, Y_test)
    train_CNN(X_train, Y_train, X_test, Y_test)
    

    


# if __name__ == "__main__":
#     main()

# 数据检测
data_dir = './data'

# 加载数据
X_train, Y_train, X_test, Y_test = load_data(data_dir)

print(f"训练数据: {X_train.shape}, 训练标签: {Y_train.shape}")
print(f"测试数据: {X_test.shape}, 测试标签: {Y_test.shape}")

















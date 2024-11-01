import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

#path
train_path = 'material/mnist_01_train.csv'
test_path = 'material/mnist_01_test.csv'

def load_data(train_path, test_path):
    # 加载数据，第一行为标签，略过
    train_data = pd.read_csv('material/mnist_01_train.csv', header=None, skiprows=1)
    test_data = pd.read_csv('material/mnist_01_test.csv', header=None, skiprows=1)
    
    # 将标签与特征分离，第一列为标签，后均为数据
    X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]
    
    # 归一化特征，收敛更快
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# 使用不同核函数的SVM训练模型
def train_svm(kernel, X_train, y_train, X_test, y_test):
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }



# 手动实现带有 hinge loss 和 cross-entropy loss 的线性分类模型
# 使用SGDClassifier（支持不同的loss）
def train_linear_model(loss, X_train, y_train, X_test, y_test):
    model = SGDClassifier(loss=loss, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def main():
    #load data
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    
    # SVM with linear kernel
    print("SVM with linear kernel:")
    results_linear_svm = train_svm('linear', X_train, y_train, X_test, y_test)
    print(results_linear_svm)

    # SVM with RBF kernel
    print("SVM with RBF kernel:")
    results_rbf_svm = train_svm('rbf', X_train, y_train, X_test, y_test)
    print(results_rbf_svm)

    # Linear model with hinge loss
    print("Linear model with hinge loss:")
    results_hinge = train_linear_model('hinge', X_train, y_train, X_test, y_test)
    print(results_hinge)

    # Linear model with cross-entropy loss 
    print("Linear model with cross-entropy loss:")
    results_log = train_linear_model('log_loss', X_train, y_train, X_test, y_test)
    print(results_log)

if __name__ == '__main__':
    main()

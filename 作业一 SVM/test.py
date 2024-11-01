import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# 加载数据
train_data = pd.read_csv('material/mnist_01_train.csv', header=None, skiprows=1)
test_data = pd.read_csv('material/mnist_01_test.csv', header=None, skiprows=1)

# 将标签与特征分离
X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

# 归一化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用不同核函数的SVM训练模型
def train_svm(kernel):
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# SVM with linear kernel
print("SVM with linear kernel:")
results_linear_svm = train_svm('linear')
print(results_linear_svm)

# SVM with RBF kernel
print("SVM with RBF kernel:")
results_rbf_svm = train_svm('rbf')
print(results_rbf_svm)

# 手动实现带有 hinge loss 和 cross-entropy loss 的线性分类模型
# 使用SGDClassifier（支持不同的loss）
def train_linear_model(loss):
    model = SGDClassifier(loss=loss, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# Linear model with hinge loss
print("Linear model with hinge loss:")
results_hinge = train_linear_model('hinge')
print(results_hinge)

# Linear model with cross-entropy loss (logistic regression)
print("Linear model with cross-entropy loss:")
results_log = train_linear_model('log_loss')
print(results_log)


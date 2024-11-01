# 2024Fall机器学习与数据挖掘 ——作业一

学号：22336084

姓名：胡舸耀

## 一、实验环境

python：3.10.13

编译器：vscode

## 二、实验要求

1) 考虑两种不同的核函数：i) 线性核函数; ii) 高斯核函数
2) 可以直接调用现成 SVM 软件包来实现
3) 手动实现采用 hinge loss 和 cross-entropy loss 的线性分类模型，并比较它们的优劣

## 三、实验原理

支持向量机（Support vector Machine，SVM）是一种经典的监督学习算法，用于解决二分类和多分类问题。其核心思想是通过在特征空间中找到一个最优的超平面来进行分类，并且间隔最大。

在这个问题中，我们可以看作用一条线（超平面）来将数据分为两类，因为数据不一定可以用单条直线将数据分为两类，我们就可以利用核函数，将平面的输入数据转换到特征空间，再利用超平面将数据分为两类，并且间隔最大。

在二维空间中有一直线为:

$$
y=ax+b
$$

令$x$为$x_1$，$y$为$x_2$，移项得：

$$
ax_1-x_2+b=0
$$

向量化为：

$$
\omega^TX+\gamma=0
$$

$$
\omega=[\omega_1,\omega_2]^T \quad X=[x_1,x_2]^T
$$

在这里$\omega_1=a\quad \omega_2=-1$，此时$\omega$和原来$a$所代表的斜率垂直，即为直线的法向量，进一步拓展到多维中，此时：

$$
\omega=[\omega_1,\omega_2,···,\omega_n]^T \quad X=[x_1,x_2,···,x_n]^T
$$

这就是我们要求超平面的方程

为了确定$\omega$中的参数，我们要求出最大间隔时的参数值，参考直线中点到线的距离，间隔$d$有：

$$
d=\frac {|\omega^Tx+\gamma|}{||\omega||}
$$

我们的目的是为了找出一个分类效果好的超平面作为分类器。分类器的好坏的评定依据是分类间隔$W=2d$的大小，即分类间隔$W$越大，我们认为这个超平面的分类效果越好。此时，求解超平面的问题就变成了求解分类间隔$W$最大化的为题。W的最大化也就是$d$最大化的。所以SVM就是要找出最大的$d$时，各个参数的值。

## 四、实验过程

### 1.导入数据

我们用 `pandas`中的 `read_csv`函数进行数据读取，经过查看数据发现，第一行为列信息介绍，读取时跳过第一行。第一列为标签，之后均为数据，将标签与数据分离。

同时在处理时对数据进行标准化，即将每个特征缩放到均值为 0、方差为 1 的分布。使数据分布均匀且计算时收敛更快。

```python
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
```

### 2.采用不同核函数的SVM模型

 **线性核 (Linear Kernel)** ：

* 适用于线性可分的数据。
* 计算简单，执行速度快。
* 在高维数据（如文本分类）中特别有效，因为线性边界可以很好地划分样本。

 **RBF 核 (Radial Basis Function Kernel)** ：

* 非线性核，适用于线性不可分的数据。
* 可以将数据映射到高维空间，使得在该空间内的边界呈现更复杂的形状。
* 更灵活，能适应数据的复杂模式，但计算复杂度更高。

在实验中直接调用SVM包进行训练，选择不同核函数即可。

```python
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
```

### 3.手动实现采用 hinge loss 和 cross-entropy loss 的线性分类模型

**Hinge Loss** （合页损失）：

* 主要用于支持向量机 (SVM) 的分类任务。
* 损失函数形式为：

  $$
  hinge\_loss=max⁡(0,1−y⋅f(x))
  $$

  其中 y 是标签， f(x) 是预测得分。
* 该损失希望让每个样本在超平面两侧的间隔至少为 1，否则会产生损失。
* 适合处理数据类别较为分明的情况，即需要找到边界的分类任务，通常适合二分类问题。

 **Cross-Entropy Loss** （交叉熵损失）：

* 主要用于逻辑回归以及神经网络的分类任务。
* 损失函数形式为：

  $$
  \text{cross\_entropy\_loss} = - (y \cdot \log(p) + (1 - y) \cdot \log(1 - p))
  $$

  其中 p 是模型对正类的预测概率。
* 该损失函数通过最大化预测的概率，使得模型更加关注正确分类的概率。
* 适合概率输出的场景，能衡量模型对不同类的偏好，在多分类和不平衡数据中较有效。

实验中通过 `SGDClassifier` 实现了带有不同损失函数的线性分类器。`hinge` 损失用于 SVM 样式的分类器，而 `log_loss` 用于逻辑回归。两者的训练和预测方法相似，唯一的区别是优化目标不同。

```python
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
```

### 4.主函数

实现了函数调用以及结果的输出

```python
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
```

## 五、实验结果与分析

实验中的模型采用了以下四个指标进行性能评估：

* **准确率 (Accuracy)** ：预测正确的样本数占总样本数的比例。
* **精确率 (Precision)** ：预测为正类的样本中实际为正类的比例。
* **召回率 (Recall)** ：实际为正类的样本中预测为正类的比例。
* **F1 值 (F1 Score)** ：精确率和召回率的调和平均，兼顾两者

运行代码，终端输出如下：

![1730470626139](image/实验报告/1730470626139.png)

### 1.不同核函数的模型性能比较

SVM with linear kernel:

{'accuracy': 0.9995271867612293, 'precision': 0.9991197183098591, 'recall': 1.0, 'f1': 0.9995596653456628}

SVM with RBF kernel:

{'accuracy': 0.9962174940898345, 'precision': 0.9991142604074402, 'recall': 0.9938325991189427, 'f1': 0.9964664310954064}

可以看到线性核的四个指标均高于高斯核指标（因为分类较简单，准确率均较高）

这个结果说明了数据是更加偏向于线性可分的，不需要较复杂的数据转化。

### 2.采用 hinge loss 线性分类模型和 cross-entropy loss 线性分类模型比较

Linear model with hinge loss:

{'accuracy': 0.9976359338061466, 'precision': 0.9982363315696648, 'recall': 0.9973568281938326, 'f1': 0.99779638607316}

Linear model with cross-entropy loss:

{'accuracy': 0.9981087470449173, 'precision': 0.9982378854625551, 'recall': 0.9982378854625551, 'f1': 0.9982378854625551}

可以看到交叉熵损失函数的性能表现优于合页损失。

这样的原因可能是数据分布不均匀或者噪声数据较多导致的。

## 六、实验讨论

到这里，我们已经实现了我们需要达到的实验要求。

因为实验数据简单的缘故，模型表现准确率较高，不能很好的体现不同模型之间的差距。

在数据处理阶段，尝试过将数据进行随机种子分配，进行随机打乱：

```python
from sklearn.utils import shuffle

def load_data(train_path, test_path):
    # 加载数据，第一行为标签，略过
    train_data = pd.read_csv(train_path, header=None, skiprows=1)
    test_data = pd.read_csv(test_path, header=None, skiprows=1)
  
    # 将标签与特征分离，第一列为标签，后均为数据
    X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]
  
    # 随机打乱训练数据
    X_train, y_train = shuffle(X_train, y_train, random_state=42)# 种子为42
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
  
    # 归一化特征，收敛更快
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
```

但是实验表现并没有较大差异，故在实验步骤中没有进行打乱展示。

在实验中我们从底层逻辑讲解了SVM的原理，以及在代码模型上的实现，同时对比线性分类模型，让我们更好的去理解了基础机器学习的过程以及方法。当然，在实验过程中不可避免遇到问题，例如在导入数据进行训练时，我们没有查看数据，而是直接查看数据文档中的解释的话，会忘记第一行的数据错误，直接导入训练，这样在后面SVM模型中训练会产生无法将 `string`数据转化为 `float`数据的错误；同时在线性分类模型中，因为不同版本的原因，交叉熵损失函数所对应的标签不同，不同版本 `sklearn`中分别为 `log`和 `log_loss`。不过在排查错误后还是顺利的进行了实验得到了实验结果，不过可能是数据分布过于简单或者是数据分类较为简单等原因，模型性能表现均极好，无法体现不同模型之间的差异以及特点。

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 动态定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        """
        动态定义 MLP 模型。
        
        参数:
        input_size (int): 输入层大小（如 CIFAR-10 的 3072）。
        hidden_layers (list): 每层隐藏层神经元数量（如 [128, 64] 表示两层，分别有 128 和 64 个神经元）。
        num_classes (int): 输出类别数量（如 CIFAR-10 的 10）。
        """
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # 使用 ReLU 激活函数
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))  # 输出层
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model_with_tracking(model, train_loader, test_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    model.train()
    test_accuracies = []  # 记录每个 epoch 的测试准确率

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 测试模型，记录测试集准确率
        accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(accuracy)

        # 打印每个 epoch 的损失和准确率
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

    return test_accuracies  # 返回每个 epoch 的测试准确率

# 模型评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def train_MLP_with_optimizers(X_train, Y_train, X_test, Y_test):
    # 超参设置
    lr = 0.001
    batch_size = 64
    num_epochs = 100
    num_classes = 10
    input_size = 32 * 32 * 3  # CIFAR-10 图像展平为 3072
    hidden_layer_config = [256, 128, 64, 32]  # 固定层配置

    # 转换为 Tensor 并创建 DataLoader
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                    torch.tensor(Y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                   torch.tensor(Y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 定义优化器字典
    optimizers = {
        "SGD": lambda model: optim.SGD(model.parameters(), lr=lr),
        "SGD Momentum": lambda model: optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "Adam": lambda model: optim.Adam(model.parameters(), lr=lr)
    }

    all_accuracies = {}  # 保存每种优化器的准确率变化

    # 运行每种优化器
    for optimizer_name, optimizer_fn in optimizers.items():
        print(f"\nTraining with {optimizer_name}...")
        model = MLP(input_size, hidden_layer_config, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer_fn(model)

        # 训练模型并记录准确率
        accuracies = train_model_with_tracking(model, train_loader, test_loader, criterion, optimizer, device, num_epochs)
        all_accuracies[optimizer_name] = accuracies

    # 绘制准确率变化曲线
    plt.figure(figsize=(12, 6))
    for optimizer_name, accuracies in all_accuracies.items():
        plt.plot(range(1, num_epochs + 1), accuracies, label=optimizer_name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy Progression for Different Optimizers")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()
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

# 模型训练函数（与之前类似）
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

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

# 实验运行函数
def run_experiments(hidden_layer_configs, train_loader, test_loader, input_size, num_classes, device, num_epochs=10, lr=0.01):
    results = []
    for config in hidden_layer_configs:
        print(f"\nTraining MLP with layers: {config}")
        model = MLP(input_size, config, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        train_model(model, train_loader, criterion, optimizer, device, num_epochs)

        # 测试模型
        accuracy = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy with layers {config}: {accuracy:.4f}")
        results.append((config, accuracy))
    return results

# 训练MLP模型生成结果
def train_MLP(X_train, Y_train, X_test, Y_test):
    # 超参设置
    lr = 0.001
    batch_size = 64
    num_epochs = 50
    num_classes = 10
    input_size = 32 * 32 * 3  # CIFAR-10 图像展平为 3072
    # 隐藏层配置列表
    hidden_layer_configs = [
        [128],
        [64, 32],
        [128,64],
        [256, 128],
        [128,64, 32],
        [256, 128, 64],
        [512, 256,128],
        [256, 128, 64, 32],
        [512, 256, 128, 64],
        [1024, 512, 256, 128],
        [512, 256, 128, 64, 32],
        [1024, 512, 256, 128, 64]
    ]

     # 转换为 Tensor 并创建 DataLoader
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                    torch.tensor(Y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                   torch.tensor(Y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    # 运行实验
    results = run_experiments(hidden_layer_configs, train_loader, test_loader, input_size, num_classes, device, num_epochs, lr)

    # 可视化结果
    configs = [str(config) for config, _ in results]
    accuracies = [accuracy for _, accuracy in results]
    plt.figure(figsize=(10, 6))
    plt.barh(configs, accuracies, color='skyblue')
    plt.xlabel("Accuracy")
    plt.ylabel("Hidden Layer Configuration")
    plt.title("MLP Performance with Different Hidden Layer Configurations")
    plt.grid(axis='x')
    plt.show()

def train_MLP_with_optimizers(X_train, Y_train, X_test, Y_test):
    # 超参设置
    lr = 0.001
    batch_size = 64
    num_epochs = 50
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
    print(device)

    # 定义优化器字典
    optimizers = {
        "SGD": lambda model: optim.SGD(model.parameters(), lr=lr),
        "SGD Momentum 0.9": lambda model: optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "SGD Momentum 0.8": lambda model: optim.SGD(model.parameters(), lr=lr, momentum=0.8),
        "Adam": lambda model: optim.Adam(model.parameters(), lr=lr)
    }

    results = []

    # 运行每种优化器
    for optimizer_name, optimizer_fn in optimizers.items():
        print(f"\nTraining with {optimizer_name}...")
        model = MLP(input_size, hidden_layer_config, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer_fn(model)

        # 训练模型
        train_model(model, train_loader, criterion, optimizer, device, num_epochs)

        # 测试模型
        accuracy = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy with {optimizer_name}: {accuracy:.4f}")
        results.append((optimizer_name, accuracy))

    # 可视化结果
    optimizer_names = [name for name, _ in results]
    accuracies = [accuracy for _, accuracy in results]
    plt.figure(figsize=(8, 5))
    plt.bar(optimizer_names, accuracies, color=['skyblue', 'lightgreen', 'coral'])
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy")
    plt.title("MLP Performance with Different Optimizers")
    plt.ylim(0, 1)  # 设置准确率范围
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
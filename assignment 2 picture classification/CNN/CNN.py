import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, conv_layers=2, filters=32, use_pooling=True):
        super(LeNet, self).__init__()
        self.use_pooling = use_pooling
        self.conv_layers = conv_layers

        # 构建卷积部分
        layers = []
        in_channels = input_channels
        for i in range(conv_layers):
            layers.append(nn.Conv2d(in_channels, filters * (2 ** i), kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if use_pooling:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = filters * (2 ** i)

        self.feature_extractor = nn.Sequential(*layers)

        # 动态计算全连接层输入大小
        self.feature_dim = self._get_feature_dim(input_channels, 32, 32)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def _get_feature_dim(self, channels, height, width):
        # 通过一层一层的模拟前向传播计算特征展平维度
        x = torch.zeros(1, channels, height, width)  # 生成一个虚拟输入
        x = self.feature_extractor(x)
        return x.numel()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

# 模型训练与评估
def train_and_evaluate(model, train_loader, test_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_accs, test_accs = [], []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_accs.append(train_acc)

        # 测试阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = correct / total
        test_accs.append(test_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    return train_accs, test_accs

def train_CNN(X_train, Y_train, X_test, Y_test):
    # 超参设置
    lr = 0.0005
    batch_size = 32
    num_epochs = 20
    num_classes = 10
    # 定义实验配置
    experiments = [
        #{"conv_layers": 2, "filters": 64, "use_pooling": True},
        #{"conv_layers": 3, "filters": 32, "use_pooling": True},
        {"conv_layers": 3, "filters": 64, "use_pooling": True},
        #{"conv_layers": 3, "filters": 64, "use_pooling": False},
        #{"conv_layers": 4, "filters": 64, "use_pooling": True},
        {"conv_layers": 3, "filters": 128, "use_pooling": True},
        #{"conv_layers": 5, "filters": 64, "use_pooling": True},
    ]
    

    # 转换为 Tensor 并创建 DataLoader
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                    torch.tensor(Y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                   torch.tensor(Y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    results = {}

    for config in experiments:
        print(f"\nTraining with config: {config}")
        model = LeNet(num_classes=num_classes, **config)
        train_accs, test_accs = train_and_evaluate(model, train_loader, test_loader, num_epochs, lr, device)
        results[str(config)] = (train_accs, test_accs)

    # 可视化不同模型的准确率对比
    plt.figure(figsize=(10, 6))
    for config, (train_accs, test_accs) in results.items():
        plt.plot(range(1, num_epochs + 1), test_accs, label=f"Config: {config}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Comparison Across Models")
    plt.legend()
    plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Softmax 分类器模型定义
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# 模型训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    epoch_losses = []  # 记录每个 epoch 的平均损失
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
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # # 绘制损失曲线
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, num_epochs + 1), epoch_losses, label="Training Loss", marker='o')
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Training Loss Over Epochs")
    # plt.legend()
    # plt.grid()
    # plt.show()

# 模型评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # 使用准确率作为评估指标
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# 模型运行实现
def train_softmax(X_train, Y_train, X_test, Y_test):
    # 超参设置
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.002
    num_classes = 10
    input_size = 32 * 32 * 3  # CIFAR-10 图像展平为 3072

    # 转换为 Tensor 并创建 DataLoader
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                    torch.tensor(Y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                   torch.tensor(Y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device) # 显示使用cpu或cuda
    model = SoftmaxClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # 评估模型
    evaluate_model(model, test_loader, device)